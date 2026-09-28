"""CLI for the fast, resumable candidate generator.

Improvements vs the original run_blocking.py
============================================
1. RESUMABLE: output is written as numbered shards (batch_000000.parquet…)
   in a staging directory. A JSON progress file tracks the last completed
   S1 row. A restart skips completed shards and merges at the end.

2. NO OUTPUT FILE DELETED AT STARTUP: the original unlinked the output
   before starting, losing all progress. Now shards accumulate safely.

3. INDEX CACHE: source mtime is checked; if unchanged, the precomputed
   numpy index is loaded from cache_dir instead of being rebuilt.
   A rebuild that took 20 min is now a 30-second load on restart.

4. RECOMMENDED FAST DEFAULTS:
     --top-k-per-source 40
     --prefilter-limit-per-source 300
     --max-postings-per-key 1000
     --max-postal-candidates 500
     --max-number-candidates 300
     --minhash-permutations 8
     --minhash-band-size 4
   These match the guidance in the analysis. Tune upward if recall is low.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TextIO

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.blocking.advanced_blocking import (
    BLOCKING_COLUMNS,
    CANDIDATE_COLUMNS,
    FastBlockingConfig,
    FastMultiSourceBlocker,
    score_candidate_recall,
)


# ──────────────────────────────────────────────────────────────────
#  I/O helpers
# ──────────────────────────────────────────────────────────────────
def read_source(path: str) -> pd.DataFrame:
    return pd.read_parquet(path, columns=BLOCKING_COLUMNS)


def source_mtime(path: str) -> float:
    return os.path.getmtime(path)


def read_id_filter(path: str | None) -> set[str] | None:
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        return {line.strip() for line in fh if line.strip()}


def iter_source1_batches(path: str, batch_size: int,
                         id_filter: set[str] | None = None,
                         skip_rows: int = 0):
    parquet = pq.ParquetFile(path)
    rows_seen = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=BLOCKING_COLUMNS):
        frame = batch.to_pandas()
        if skip_rows and rows_seen + len(frame) <= skip_rows:
            rows_seen += len(frame)
            continue
        if skip_rows and rows_seen < skip_rows:
            frame = frame.iloc[skip_rows - rows_seen:]
        rows_seen += len(frame)
        if id_filter is not None:
            frame = frame[frame["entity_id"].isin(id_filter)]
        if not frame.empty:
            yield frame


# ──────────────────────────────────────────────────────────────────
#  Shard-based resumable writer
# ──────────────────────────────────────────────────────────────────
class ShardWriter:
    """Write candidate rows as numbered parquet shards for resumability.

    Progress is tracked in <staging_dir>/progress.json:
      {"completed_shards": N, "completed_s1_rows": M}
    A restart reads this file and skips the first N shards.
    """

    def __init__(self, staging_dir: Path):
        self.staging_dir = staging_dir
        staging_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = staging_dir / "progress.json"
        self._shard_no: int = 0
        self._completed_rows: int = 0

        if self.progress_path.is_file():
            prog = json.loads(self.progress_path.read_text(encoding="utf-8"))
            self._shard_no     = prog.get("completed_shards", 0)
            self._completed_rows = prog.get("completed_s1_rows", 0)

    @property
    def resume_from_row(self) -> int:
        return self._completed_rows

    def write_shard(self, candidates: pd.DataFrame, s1_rows_processed: int) -> None:
        """Write one shard and update progress."""
        shard_path = self.staging_dir / f"batch_{self._shard_no:06d}.parquet"
        table = pa.Table.from_pandas(candidates, preserve_index=False)
        pq.write_table(table, shard_path, compression="snappy")
        self._completed_rows += s1_rows_processed
        self._shard_no += 1
        self.progress_path.write_text(
            json.dumps({
                "completed_shards": self._shard_no,
                "completed_s1_rows": self._completed_rows,
            }),
            encoding="utf-8",
        )

    def merge_to(self, output_path: Path) -> None:
        """Concatenate all shards into the final output parquet."""
        shards = sorted(self.staging_dir.glob("batch_*.parquet"))
        if not shards:
            pd.DataFrame(columns=CANDIDATE_COLUMNS).to_parquet(output_path, index=False)
            return
        import duckdb
        shard_glob = str(self.staging_dir / "batch_*.parquet").replace("\\", "/")
        out = str(output_path).replace("\\", "/")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect()
        con.execute(f"""
        COPY (
          SELECT * FROM read_parquet('{shard_glob}')
          ORDER BY source1_entity_id, blocking_score DESC
        ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
        """)
        con.close()
        print(f"  merged {len(shards)} shards → {output_path}", flush=True)


# ──────────────────────────────────────────────────────────────────
#  Argument parsing
# ──────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate S1-to-S2/S3 candidate pairs (fast, resumable)."
    )
    parser.add_argument("--source1", required=True)
    parser.add_argument("--source2", required=True)
    parser.add_argument("--source3", required=True)
    parser.add_argument("--output",  required=True,
                        help="Final merged candidate parquet output.")
    parser.add_argument("--staging-dir", default=None,
                        help="Temp dir for shards + progress file. "
                             "Defaults to <output>.staging/")
    parser.add_argument("--cache-dir", default=None,
                        help="Dir to persist/load source index caches.")
    parser.add_argument("--candidate-pairs-tsv",
                        help="Optional challenge-format candidate_pairs.tsv.")
    parser.add_argument("--s1-id-filter",
                        help="Optional text file of Source 1 IDs to process.")
    parser.add_argument("--ground-truth",
                        help="Optional TSV to measure recall after blocking.")
    parser.add_argument("--batch-size", type=int, default=5_000)
    # ── Fast-tuned defaults (half the original resource usage) ─────────
    parser.add_argument("--top-k-per-source",             type=int, default=40)
    parser.add_argument("--prefilter-limit-per-source",   type=int, default=300)
    parser.add_argument("--max-postings-per-key",         type=int, default=1_000)
    parser.add_argument("--max-postal-candidates",        type=int, default=500)
    parser.add_argument("--max-number-candidates",        type=int, default=300)
    parser.add_argument("--minhash-permutations",         type=int, default=8)
    parser.add_argument("--minhash-band-size",            type=int, default=4)
    parser.add_argument("--min-final-score",              type=float, default=0.35)
    parser.add_argument("--min-name-similarity",          type=float, default=0.22)
    parser.add_argument("--multi-evidence-threshold",     type=int, default=2)
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    output_path  = Path(args.output)
    staging_dir  = Path(args.staging_dir) if args.staging_dir else output_path.with_suffix(".staging")
    cache_dir    = Path(args.cache_dir) if args.cache_dir else None

    config = FastBlockingConfig(
        top_k_per_source             = args.top_k_per_source,
        prefilter_limit_per_source   = args.prefilter_limit_per_source,
        max_postings_per_key         = args.max_postings_per_key,
        max_postal_candidates        = args.max_postal_candidates,
        max_number_candidates        = args.max_number_candidates,
        minhash_permutations         = args.minhash_permutations,
        minhash_band_size            = args.minhash_band_size,
        min_final_score              = args.min_final_score,
        min_name_similarity          = args.min_name_similarity,
        multi_evidence_threshold     = args.multi_evidence_threshold,
    )

    # ── Load sources ───────────────────────────────────────────────
    print("Loading source tables…", flush=True)
    source2 = read_source(args.source2)
    source3 = read_source(args.source3)

    # ── Build / load cached indexes ────────────────────────────────
    blocker = FastMultiSourceBlocker(
        source2, source3, config,
        cache_dir=cache_dir,
        source2_mtime=source_mtime(args.source2),
        source3_mtime=source_mtime(args.source3),
    )
    del source2, source3  # free RAM before S1 loop

    # ── Shard writer (resumable) ───────────────────────────────────
    shard_writer = ShardWriter(staging_dir)
    skip_rows    = shard_writer.resume_from_row
    id_filter    = read_id_filter(args.s1_id_filter)

    if skip_rows > 0:
        print(f"Resuming from S1 row {skip_rows:,} "
              f"(shard {shard_writer._shard_no})", flush=True)

    # ── TSV writer ─────────────────────────────────────────────────
    tsv_handle: TextIO | None = None
    if args.candidate_pairs_tsv:
        tsv_path = Path(args.candidate_pairs_tsv)
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if skip_rows > 0 else "w"
        tsv_handle = tsv_path.open(mode, encoding="utf-8", newline="")
        if skip_rows == 0:
            tsv_handle.write("source1_entity_id\tcandidate_entity_ids\n")

    processed = skip_rows
    try:
        for chunk in iter_source1_batches(
            args.source1, args.batch_size, id_filter, skip_rows
        ):
            s1_ids = chunk["entity_id"].astype(str).tolist()
            candidates = blocker.generate_chunk(chunk)
            shard_writer.write_shard(candidates, len(chunk))

            if tsv_handle is not None and not candidates.empty:
                grouped = (
                    candidates
                    .sort_values(["source1_entity_id", "blocking_score"],
                                 ascending=[True, False])
                    .groupby("source1_entity_id")["candidate_entity_id"]
                    .apply(lambda vs: ",".join(dict.fromkeys(vs.astype(str))))
                    .to_dict()
                )
                for sid in s1_ids:
                    tsv_handle.write(f"{sid}\t{grouped.get(sid, '')}\n")

            processed += len(chunk)
            print(
                f"processed {processed:,} S1 rows | "
                f"batch candidates={len(candidates):,} | "
                f"shard={shard_writer._shard_no}",
                flush=True,
            )
    finally:
        if tsv_handle:
            tsv_handle.close()

    # ── Merge shards → final output ────────────────────────────────
    print("Merging shards…", flush=True)
    shard_writer.merge_to(output_path)
    print(f"wrote {output_path}", flush=True)

    # ── Optional recall evaluation ─────────────────────────────────
    if args.ground_truth:
        import pandas as pd
        gt = pd.read_csv(args.ground_truth, sep="\t", dtype=str, keep_default_na=False)
        candidates_all = pq.read_table(output_path).to_pandas()
        recall_metrics = score_candidate_recall(candidates_all, gt)
        print(json.dumps(recall_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
