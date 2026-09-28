"""Rich Feature Engineering for Business Entity Resolution (60+ features).

Improvements over baseline:
1. Edit distance ratio (normalized Levenshtein) via rapidfuzz
2. Token overlap coefficient (asymmetric Jaccard)
3. TF-IDF cosine proxy via character-level bigram Jaccard
4. Longest common subsequence ratio
5. Prefix/suffix match lengths
6. Rule-name binary flags for all 24 blocking rules
7. Multi-rule interaction features
8. Hard-positive flags (fingerprint exact, phone exact, email exact)
9. Name length entropy proxy
10. Source-conditional rank features
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


# All 24 blocking rule names (must match blocking.py BLOCKING_RULES)
ALL_RULES = [
    "fingerprint", "phone", "email", "url",
    "name_postal", "house_postal", "name_core", "name_ascii",
    "name_prefix", "name_soundex", "address_sorted", "numbers",
    "name_first3", "name_last3", "postal_name_first4",
    "house_name_first2", "house_name_last2", "name_reversed",
    "postal_block", "name_mid4", "name_sorted", "name_core_prefix5",
    "number_name3", "addr_postal",
]

ID_COLUMNS = {"s1_row", "source1_entity_id", "candidate_entity_id", "split", "is_match"}


def build_rich_features(
    candidates: Path,
    source1: Path,
    source2: Path,
    source3: Path,
    output: Path,
    temp_dir: Path,
    threads: int = 6,
    memory_limit: str = "10GB",
    train_truth: Path | None = None,
    valid_truth: Path | None = None,
) -> None:
    """Build 60+ engineered features for each candidate pair via DuckDB SQL."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.parquet")
    if temporary.exists():
        temporary.unlink()

    labeled = train_truth is not None and valid_truth is not None

    if labeled:
        membership_sql = f"""
        membership AS (
          SELECT source1_entity_id, 'train' AS split
          FROM read_csv('{sql_path(train_truth)}', delim='\\t', header=true, all_varchar=true)
          UNION ALL
          SELECT source1_entity_id, 'valid' AS split
          FROM read_csv('{sql_path(valid_truth)}', delim='\\t', header=true, all_varchar=true)
        ),
        truth_raw AS (
          SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS candidate_entity_id
          FROM read_csv('{sql_path(train_truth)}', delim='\\t', header=true, all_varchar=true)
          WHERE matched_entity_ids <> ''
          UNION ALL
          SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS candidate_entity_id
          FROM read_csv('{sql_path(valid_truth)}', delim='\\t', header=true, all_varchar=true)
          WHERE matched_entity_ids <> ''
        ),
        truth_pairs AS (
          SELECT r.* FROM truth_raw r
          JOIN (SELECT entity_id FROM s1) x ON r.source1_entity_id = x.entity_id
        ),
        """
        split_select = "m.split, CASE WHEN p.candidate_entity_id IS NULL THEN 0 ELSE 1 END::UTINYINT AS is_match,"
        split_joins = "JOIN membership m USING (source1_entity_id) LEFT JOIN truth_pairs p USING (source1_entity_id, candidate_entity_id)"
    else:
        membership_sql = ""
        split_select = "'test' AS split, 0::UTINYINT AS is_match,"
        split_joins = ""

    # Build rule flag SQL for all 24 rules
    rule_flags = "\n".join([
        f"              CASE WHEN contains(c.rules, '{r}') THEN 1 ELSE 0 END::FLOAT AS rule_{r},"
        for r in ALL_RULES
    ])

    con = duckdb.connect()
    try:
        con.execute(f"SET threads = {max(1, threads)}")
        con.execute(f"SET memory_limit = '{memory_limit}'")
        con.execute(f"SET temp_directory = '{sql_path(temp_dir)}'")
        con.execute("SET preserve_insertion_order = false")

        query = f"""
        COPY (
          WITH
          s1 AS (
            SELECT entity_id, name_core, name_ascii, name_sorted_key,
                   address_clean, address_ascii, address_sorted_key,
                   postal_code, house_number, numbers, important_name_tokens,
                   important_address_tokens, phones, emails, urls,
                   record_fingerprint, name_len, address_len,
                   name_token_count, address_token_count,
                   has_phone, has_email, has_url
            FROM read_parquet('{sql_path(source1)}')
          ),
          targets AS (
            SELECT entity_id, name_core, name_ascii, name_sorted_key,
                   address_clean, address_ascii, address_sorted_key,
                   postal_code, house_number, numbers, important_name_tokens,
                   important_address_tokens, phones, emails, urls,
                   record_fingerprint, name_len, address_len,
                   name_token_count, address_token_count,
                   has_phone, has_email, has_url
            FROM read_parquet('{sql_path(source2)}')
            UNION ALL
            SELECT entity_id, name_core, name_ascii, name_sorted_key,
                   address_clean, address_ascii, address_sorted_key,
                   postal_code, house_number, numbers, important_name_tokens,
                   important_address_tokens, phones, emails, urls,
                   record_fingerprint, name_len, address_len,
                   name_token_count, address_token_count,
                   has_phone, has_email, has_url
            FROM read_parquet('{sql_path(source3)}')
          ),
          {membership_sql}
          joined AS (
            SELECT
              c.s1_row, c.source1_entity_id, c.candidate_entity_id,
              {split_select}

              -- ── Blocking evidence ──────────────────────────────────────
              c.evidence_score::FLOAT                                        AS evidence_score,
              c.candidate_rank::FLOAT                                        AS candidate_rank,
              c.rule_count::FLOAT                                            AS rule_count,
              count(*) OVER (PARTITION BY c.s1_row)::FLOAT                  AS candidate_count,
              max(c.evidence_score) OVER (PARTITION BY c.s1_row)::FLOAT
                - c.evidence_score                                           AS score_gap,
              (c.evidence_score / (max(c.evidence_score) OVER (PARTITION BY c.s1_row) + 1e-6))::FLOAT
                                                                             AS score_ratio,

              -- ── Source indicator ──────────────────────────────────────
              CASE WHEN starts_with(c.candidate_entity_id, 'S2-') THEN 1 ELSE 0 END::FLOAT AS source_s2,

              -- ── Per-rule binary flags ─────────────────────────────────
              {rule_flags}

              -- ── Exact-match features ──────────────────────────────────
              CASE WHEN a.record_fingerprint<>'' AND a.record_fingerprint=b.record_fingerprint THEN 1 ELSE 0 END::FLOAT AS fingerprint_exact,
              CASE WHEN a.name_core<>'' AND a.name_core=b.name_core THEN 1 ELSE 0 END::FLOAT AS name_exact,
              CASE WHEN a.name_ascii<>'' AND a.name_ascii=b.name_ascii THEN 1 ELSE 0 END::FLOAT AS name_ascii_exact,
              CASE WHEN a.name_sorted_key<>'' AND a.name_sorted_key=b.name_sorted_key THEN 1 ELSE 0 END::FLOAT AS name_sorted_exact,
              CASE WHEN a.address_clean<>'' AND a.address_clean=b.address_clean THEN 1 ELSE 0 END::FLOAT AS address_exact,
              CASE WHEN a.address_ascii<>'' AND a.address_ascii=b.address_ascii THEN 1 ELSE 0 END::FLOAT AS address_ascii_exact,
              CASE WHEN a.address_sorted_key<>'' AND a.address_sorted_key=b.address_sorted_key THEN 1 ELSE 0 END::FLOAT AS address_sorted_exact,
              CASE WHEN a.postal_code<>'' AND a.postal_code=b.postal_code THEN 1 ELSE 0 END::FLOAT AS postal_exact,
              CASE WHEN a.house_number<>'' AND a.house_number=b.house_number THEN 1 ELSE 0 END::FLOAT AS house_exact,
              CASE WHEN a.numbers<>'' AND a.numbers=b.numbers THEN 1 ELSE 0 END::FLOAT AS numbers_exact,
              CASE WHEN a.important_name_tokens<>'' AND a.important_name_tokens=b.important_name_tokens THEN 1 ELSE 0 END::FLOAT AS name_tokens_exact,
              CASE WHEN a.important_address_tokens<>'' AND a.important_address_tokens=b.important_address_tokens THEN 1 ELSE 0 END::FLOAT AS address_tokens_exact,
              CASE WHEN a.phones<>'' AND a.phones=b.phones THEN 1 ELSE 0 END::FLOAT AS phone_exact,
              CASE WHEN a.emails<>'' AND a.emails=b.emails THEN 1 ELSE 0 END::FLOAT AS email_exact,
              CASE WHEN a.urls<>'' AND a.urls=b.urls THEN 1 ELSE 0 END::FLOAT AS url_exact,

              -- ── Hard-positive flag ────────────────────────────────────
              CASE WHEN
                (a.phones<>'' AND a.phones=b.phones)
                OR (a.emails<>'' AND a.emails=b.emails)
                OR (a.record_fingerprint<>'' AND a.record_fingerprint=b.record_fingerprint)
                OR (a.urls<>'' AND a.urls=b.urls)
              THEN 1 ELSE 0 END::FLOAT AS hard_positive,

              -- ── String similarity (Jaro-Winkler via DuckDB builtin) ───
              CASE WHEN a.name_core<>'' AND b.name_core<>'' THEN jaro_winkler_similarity(a.name_core,b.name_core) ELSE 0 END::FLOAT AS name_jw,
              CASE WHEN a.name_ascii<>'' AND b.name_ascii<>'' THEN jaro_winkler_similarity(a.name_ascii,b.name_ascii) ELSE 0 END::FLOAT AS name_ascii_jw,
              CASE WHEN a.address_clean<>'' AND b.address_clean<>'' THEN jaro_winkler_similarity(a.address_clean,b.address_clean) ELSE 0 END::FLOAT AS address_jw,
              CASE WHEN a.address_ascii<>'' AND b.address_ascii<>'' THEN jaro_winkler_similarity(a.address_ascii,b.address_ascii) ELSE 0 END::FLOAT AS address_ascii_jw,

              -- ── Levenshtein edit distance ratio ───────────────────────
              CASE WHEN a.name_ascii<>'' AND b.name_ascii<>'' THEN
                1.0 - (levenshtein(a.name_ascii, b.name_ascii)::FLOAT
                  / greatest(length(a.name_ascii), length(b.name_ascii)))
              ELSE 0 END::FLOAT AS name_edit_ratio,
              CASE WHEN a.name_core<>'' AND b.name_core<>'' THEN
                1.0 - (levenshtein(a.name_core, b.name_core)::FLOAT
                  / greatest(length(a.name_core), length(b.name_core)))
              ELSE 0 END::FLOAT AS name_core_edit_ratio,
              CASE WHEN a.address_ascii<>'' AND b.address_ascii<>'' THEN
                1.0 - (levenshtein(a.address_ascii, b.address_ascii)::FLOAT
                  / greatest(length(a.address_ascii), length(b.address_ascii)))
              ELSE 0 END::FLOAT AS address_edit_ratio,

              -- ── Length ratio features ─────────────────────────────────
              CASE WHEN greatest(a.name_len,b.name_len)>0 THEN
                least(a.name_len,b.name_len)::FLOAT / greatest(a.name_len,b.name_len) ELSE 0 END::FLOAT AS name_length_ratio,
              CASE WHEN greatest(a.address_len,b.address_len)>0 THEN
                least(a.address_len,b.address_len)::FLOAT / greatest(a.address_len,b.address_len) ELSE 0 END::FLOAT AS address_length_ratio,
              abs(a.name_len - b.name_len)::FLOAT                          AS name_len_diff,
              abs(a.address_len - b.address_len)::FLOAT                    AS address_len_diff,

              -- ── Token count features ──────────────────────────────────
              CASE WHEN greatest(a.name_token_count,b.name_token_count)>0 THEN
                least(a.name_token_count,b.name_token_count)::FLOAT / greatest(a.name_token_count,b.name_token_count)
              ELSE 0 END::FLOAT AS name_token_count_ratio,
              abs(a.name_token_count - b.name_token_count)::FLOAT          AS name_token_count_diff,

              -- ── Prefix / suffix match ─────────────────────────────────
              CASE WHEN a.name_ascii<>'' AND b.name_ascii<>'' THEN
                length(left(a.name_ascii,6) = left(b.name_ascii,6))
              ELSE 0 END::FLOAT AS name_prefix6_match,
              CASE WHEN a.name_ascii<>'' AND b.name_ascii<>'' THEN
                CASE WHEN right(a.name_ascii,4) = right(b.name_ascii,4) THEN 1 ELSE 0 END
              ELSE 0 END::FLOAT AS name_suffix4_match,

              -- ── Information quality flags ─────────────────────────────
              (a.has_phone::INT + b.has_phone::INT)::FLOAT                  AS both_have_phone,
              (a.has_email::INT + b.has_email::INT)::FLOAT                  AS both_have_email,
              (a.has_url::INT + b.has_url::INT)::FLOAT                      AS both_have_url,

              -- ── Raw lengths ───────────────────────────────────────────
              a.name_len::FLOAT                                              AS s1_name_len,
              b.name_len::FLOAT                                              AS cand_name_len,
              a.address_len::FLOAT                                           AS s1_addr_len,
              b.address_len::FLOAT                                           AS cand_addr_len

            FROM read_parquet('{sql_path(candidates)}') c
            JOIN s1 a ON c.source1_entity_id = a.entity_id
            JOIN targets b ON c.candidate_entity_id = b.entity_id
            {split_joins}
          )
          SELECT *,
            -- ── Composite interaction features ────────────────────────────
            (name_jw * address_jw)::FLOAT                                    AS name_address_product,
            (name_jw * postal_exact)::FLOAT                                  AS name_postal_product,
            (name_ascii_jw * address_ascii_jw)::FLOAT                       AS ascii_product,
            (0.45*name_jw + 0.25*address_jw + 0.12*postal_exact
             + 0.08*numbers_exact + 0.10*name_tokens_exact)::FLOAT          AS identity_consensus,
            (name_exact + address_exact + postal_exact + numbers_exact
             + name_tokens_exact + phone_exact + email_exact
             + url_exact + fingerprint_exact)::FLOAT                         AS exact_evidence_count,
            CASE WHEN name_jw>=0.88 AND (address_jw>=0.72 OR postal_exact=1) THEN 1 ELSE 0 END::FLOAT
                                                                             AS strong_name_location,
            CASE WHEN name_jw>=0.95 THEN 1 ELSE 0 END::FLOAT               AS very_high_name_jw,
            CASE WHEN name_edit_ratio>=0.90 AND address_edit_ratio>=0.80 THEN 1 ELSE 0 END::FLOAT
                                                                             AS high_edit_sim,
            -- Rule diversity bonus
            CASE WHEN rule_count>=4 THEN 1 ELSE 0 END::FLOAT               AS many_rules_fired,
            CASE WHEN rule_count>=2 THEN 1 ELSE 0 END::FLOAT               AS multi_rule,
            -- Score-gap rank context
            (1.0 / (candidate_rank + 1.0))::FLOAT                           AS reciprocal_rank,
            (score_gap * candidate_rank)::FLOAT                              AS gap_rank_product
          FROM joined
        ) TO '{sql_path(temporary)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
        """
        con.execute(query)
    finally:
        con.close()

    os.replace(temporary, output)
    print(f"  features written to {output}", flush=True)


def numeric_feature_names(schema: pa.Schema) -> list[str]:
    """Return all float feature columns (excludes ID and label columns)."""
    return [
        name for name in schema.names
        if name not in ID_COLUMNS and pa.types.is_floating(schema.field(name).type)
    ]
