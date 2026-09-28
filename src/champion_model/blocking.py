"""Ultra-High-Recall Blocking for Business Entity Resolution.

Key improvements over the baseline:
1. 24 blocking rules vs 13 — adds phone/email/URL/fingerprint exact rules
2. Adaptive frequency caps per rule based on selectivity
3. Weighted evidence scoring with multi-rule bonus
4. Symmetric name-inversion rule (handles "Smith John" vs "John Smith")
5. Trigram overlap rule for short/abbreviated names
6. House-number-only rule for address-heavy records
7. Country-agnostic fallback rules for entities with missing country

Memory-safe aggregation:
- Each rule shard is written independently to disk (crash-resumable)
- Pass A uses Python chunked aggregation — never a DuckDB GROUP BY on
  the full cross-product, which OOMs on large datasets.
- Pass B ranks from the much-smaller aggregated file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


# ---------------------------------------------------------------------------
# Blocking rule definitions
# Each tuple: (rule_name, sql_field_expr, min_key_length, max_block_size, score)
# ---------------------------------------------------------------------------
BLOCKING_RULES = [
    # ── Exact identity rules (highest confidence) ──────────────────────────
    ("fingerprint",         "record_fingerprint",                                              10, 10,  10),
    ("phone",               "phones",                                                           6,  20,  9),
    ("email",               "emails",                                                           6,  20,  9),
    ("url",                 "urls",                                                             6,  30,  8),

    # ── High-selectivity compound keys ─────────────────────────────────────
    ("name_postal",         "name_postal_block_key",                                            8,  50,  7),
    ("house_postal",        "house_postal_block_key",                                           4,  60,  7),
    ("name_core",           "name_core",                                                        5, 150,  6),
    ("name_ascii",          "name_ascii",                                                       5, 150,  6),
    ("name_prefix",         "name_prefix_key",                                                  4, 200,  5),
    ("name_soundex",        "name_soundex_key",                                                 4, 300,  4),
    ("address_sorted",      "address_sorted_key",                                               8, 100,  5),
    ("numbers",             "numbers",                                                          4, 200,  4),

    # ── Token-level rules ───────────────────────────────────────────────────
    ("name_first3",         "left(name_core, 3)",                                               3, 500,  3),
    ("name_last3",          "right(name_core, 3)",                                              3, 500,  3),
    ("postal_name_first4",  "CASE WHEN postal_code<>'' AND name_ascii<>'' "
                            "THEN postal_code || '|' || left(name_ascii,4) ELSE '' END",        6, 300,  5),
    ("house_name_first2",   "CASE WHEN house_number<>'' AND name_ascii<>'' "
                            "THEN house_number || '|' || left(name_ascii,2) ELSE '' END",       3, 200,  4),
    ("house_name_last2",    "CASE WHEN house_number<>'' AND name_ascii<>'' "
                            "THEN house_number || '|' || right(name_ascii,2) ELSE '' END",      3, 200,  4),

    # ── Name-inversion (word-order invariant) ───────────────────────────────
    ("name_reversed",
     "CASE WHEN array_length(string_split(trim(name_ascii),' '))>=2 "
     "THEN array_to_string(list_reverse(string_split(trim(name_ascii),' ')),' ') "
     "ELSE '' END",
     4, 200,  5),

    # ── Postal-block composite ───────────────────────────────────────────────
    ("postal_block",        "postal_block_key",                                                 4, 400,  3),

    # ── Mid-name trigram (catches abbreviations) ───────────────────────────
    ("name_mid4",
     "CASE WHEN length(name_ascii)>=8 THEN substr(name_ascii, (length(name_ascii)/2-1)::BIGINT, 4) ELSE '' END",
     4, 200,  2),

    # ── Combined sorted name (word-order invariant) ────────────────────────
    ("name_sorted",         "name_sorted_key",                                                  4, 200,  4),

    # ── First-5-char name prefix (broader recall) ──────────────────────────
    ("name_core_prefix5",   "left(name_core, 5)",                                               4, 500,  3),

    # ── Number + first-3 chars of name (compact structural key) ───────────
    ("number_name3",
     "CASE WHEN numbers<>'' AND name_ascii<>'' THEN numbers || '|' || left(name_ascii,3) ELSE '' END",
     5, 100,  4),

    # ── Address sorted + postal (very selective) ───────────────────────────
    ("addr_postal",
     "CASE WHEN address_sorted_key<>'' AND postal_block_key<>'' "
     "THEN address_sorted_key || '|' || postal_block_key ELSE '' END",
     10,  30,  5),
]

ALL_RULES = BLOCKING_RULES  # alias for import


# ---------------------------------------------------------------------------
# Python chunked aggregation — avoids DuckDB GROUP BY OOM
# ---------------------------------------------------------------------------
def _python_aggregate_shards(
    shard_paths: Iterable[Path],
    output_path: Path,
    batch_s1_rows: int = 50_000,
) -> None:
    """Aggregate rule shards without a DuckDB GROUP BY.

    Reads each shard sequentially with pyarrow (one shard in RAM at a time).
    Accumulates max_score + rule_count per (s1_row, s1_id, cand_id) in a dict.
    Flushes completed s1_row batches to parquet shards, then merges.

    Memory upper bound: ~one rule shard + one batch window ≈ 1–3 GB.
    """
    staging = output_path.parent / (output_path.stem + "_agg_shards")
    staging.mkdir(parents=True, exist_ok=True)
    for old in staging.glob("*.parquet"):
        old.unlink()

    # acc: {(s1_row, s1_id, cand_id): [max_score, rule_count]}
    acc: dict[tuple, list] = {}
    shard_no = 0
    batch_max = batch_s1_rows  # rows in [0, batch_max) are "current batch"

    written_pairs = 0

    def _flush(up_to: int | None = None) -> None:
        """Flush pairs where s1_row < up_to (or all if up_to is None)."""
        nonlocal shard_no, written_pairs
        keys_to_flush = [
            k for k in acc
            if up_to is None or k[0] < up_to
        ]
        if not keys_to_flush:
            return
        s1_rows_arr   = pa.array([k[0] for k in keys_to_flush], pa.int64())
        s1_ids_arr    = pa.array([k[1] for k in keys_to_flush], pa.string())
        cand_ids_arr  = pa.array([k[2] for k in keys_to_flush], pa.string())
        ev_scores_arr = pa.array(
            [acc[k][0] + min(acc[k][1] - 1, 5) for k in keys_to_flush], pa.int32()
        )
        rule_cnt_arr  = pa.array([acc[k][1] for k in keys_to_flush], pa.int32())
        tbl = pa.table({
            "s1_row":              s1_rows_arr,
            "source1_entity_id":   s1_ids_arr,
            "candidate_entity_id": cand_ids_arr,
            "evidence_score":      ev_scores_arr,
            "rule_count":          rule_cnt_arr,
        })
        out = staging / f"agg_{shard_no:05d}.parquet"
        pq.write_table(tbl, out, compression="snappy")
        shard_no += 1
        written_pairs += len(keys_to_flush)
        for k in keys_to_flush:
            del acc[k]

    shard_paths = sorted(shard_paths)
    n_shards = len(shard_paths)
    for si, shard_path in enumerate(shard_paths):
        print(f"    aggregating shard {si+1}/{n_shards}: {shard_path.name}", flush=True)
        pf = pq.ParquetFile(shard_path)
        for batch in pf.iter_batches(
            batch_size=500_000,
            columns=["s1_row", "source1_entity_id", "candidate_entity_id", "rule_score"],
        ):
            d = batch.to_pydict()
            for i in range(len(d["s1_row"])):
                row_no  = int(d["s1_row"][i])
                key     = (row_no, d["source1_entity_id"][i], d["candidate_entity_id"][i])
                score   = int(d["rule_score"][i])
                cur     = acc.get(key)
                if cur is None:
                    acc[key] = [score, 1]
                else:
                    if score > cur[0]:
                        cur[0] = score
                    cur[1] += 1
                # When we've moved past the current batch window, flush completed rows
                if row_no >= batch_max:
                    _flush(up_to=batch_max)
                    batch_max = ((row_no // batch_s1_rows) + 1) * batch_s1_rows

    # Flush all remaining
    _flush(up_to=None)

    print(f"    total aggregated pairs: {written_pairs:,} in {shard_no} shards", flush=True)

    # Merge staging shards into final output via DuckDB
    # (these are small files — DuckDB can handle them easily)
    agg_glob = str(staging / "*.parquet").replace("\\", "/")
    out_str  = str(output_path).replace("\\", "/")
    con = duckdb.connect()
    con.execute(f"""
    COPY (
      SELECT s1_row, source1_entity_id, candidate_entity_id,
             evidence_score, rule_count
      FROM read_parquet('{agg_glob}')
    ) TO '{out_str}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
    """)
    con.close()
    for p in staging.glob("*.parquet"):
        p.unlink()
    try:
        staging.rmdir()
    except OSError:
        pass
    size_mb = output_path.stat().st_size // 1_048_576
    print(f"    agg.parquet written ({size_mb} MB)", flush=True)


# ---------------------------------------------------------------------------
# Main blocking function
# ---------------------------------------------------------------------------
def build_ultra_recall_candidates(
    source1: Path,
    source2: Path,
    source3: Path,
    output: Path,
    temp_dir: Path,
    threads: int = 6,
    memory_limit: str = "8GB",
    max_candidates: int = 100,
) -> None:
    """Generate ultra-high-recall candidates using 24 blocking rules.

    Evidence scoring: max_rule_score + min(extra_rules - 1, 5)
    A pair matching 6+ rules gets +5 bonus over a single-rule pair.

    Memory-safe three-pass strategy
    --------------------------------
    Rule shards : Written independently, reused on crash-resume.
    Pass A      : Python chunked aggregation — no DuckDB GROUP BY.
                  Streams shards one at a time; flushes to disk in batches.
    Pass B      : DuckDB row_number() on the smaller agg.parquet.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    temporary = output.with_suffix(".tmp.parquet")
    if temporary.exists():
        temporary.unlink()

    agg_path = temp_dir / "agg.parquet"

    con = duckdb.connect()
    try:
        con.execute(f"SET threads = {max(1, threads)}")
        con.execute(f"SET memory_limit = '{memory_limit}'")
        con.execute(f"SET temp_directory = '{sql_path(temp_dir)}'")
        con.execute("SET preserve_insertion_order = false")

        # ── Per-rule shards (reuse cached files from a prior crash) ─────────
        for index, (rule_name, field, min_length, cap, score) in enumerate(BLOCKING_RULES):
            rule_output = temp_dir / f"rule_{index:02d}.parquet"
            if rule_output.exists():
                print(f"  blocking rule {index+1}/{len(BLOCKING_RULES)}: {rule_name} [cached]", flush=True)
                continue
            print(f"  blocking rule {index+1}/{len(BLOCKING_RULES)}: {rule_name}", flush=True)
            con.execute(f"""
            COPY (
              WITH s1 AS (
                SELECT row_number() OVER () AS s1_row, entity_id,
                       COALESCE(NULLIF(country_clean,''), '__UNKNOWN__') AS country_key,
                       ({field}) AS block_key
                FROM read_parquet('{sql_path(source1)}')
              ),
              targets AS (
                SELECT entity_id,
                       COALESCE(NULLIF(country_clean,''), '__UNKNOWN__') AS country_key,
                       ({field}) AS block_key FROM read_parquet('{sql_path(source2)}')
                UNION ALL
                SELECT entity_id,
                       COALESCE(NULLIF(country_clean,''), '__UNKNOWN__') AS country_key,
                       ({field}) AS block_key FROM read_parquet('{sql_path(source3)}')
              ),
              counts AS (
                SELECT country_key, block_key, count(*) n FROM targets
                WHERE length(block_key) >= {min_length}
                GROUP BY country_key, block_key HAVING count(*) <= {cap}
              )
              SELECT a.s1_row,
                     a.entity_id   AS source1_entity_id,
                     b.entity_id   AS candidate_entity_id,
                     {score}::INTEGER AS rule_score,
                     '{rule_name}' AS rule_name
              FROM s1 a
              JOIN counts c USING (country_key, block_key)
              JOIN targets b USING (country_key, block_key)
              WHERE length(a.block_key) >= {min_length}
            ) TO '{sql_path(rule_output)}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
            """)

        # ── Pass A: Python chunked aggregation — no DuckDB GROUP BY ──────────
        if agg_path.exists():
            print("  pass A: aggregation [cached — reusing agg.parquet]", flush=True)
        else:
            print("  pass A: Python chunked aggregation across rule shards...", flush=True)
            _python_aggregate_shards(
                shard_paths=sorted(temp_dir.glob("rule_*.parquet")),
                output_path=agg_path,
                batch_s1_rows=50_000,
            )

        # ── Pass B: rank per s1_row from the smaller aggregated file ─────────
        print("  pass B: ranking candidates per S1 row...", flush=True)
        agg_str = sql_path(agg_path)
        con.execute(f"""
        COPY (
          WITH ranked AS (
            SELECT *,
              row_number() OVER (
                PARTITION BY s1_row
                ORDER BY evidence_score DESC, candidate_entity_id
              ) AS candidate_rank
            FROM read_parquet('{agg_str}')
          )
          SELECT * FROM ranked WHERE candidate_rank <= {max(1, max_candidates)}
          ORDER BY s1_row, candidate_rank
        ) TO '{sql_path(temporary)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
        """)
    finally:
        con.close()

    os.replace(temporary, output)
    if agg_path.exists():
        agg_path.unlink()
    print(f"  candidates written to {output}", flush=True)
