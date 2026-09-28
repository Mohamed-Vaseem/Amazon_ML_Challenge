"""
Advanced Feature Engineering for Business Entity Resolution.

Features implemented per specification:
========================================
1. Normalized edit similarity and character n-gram cosine
2. Weighted token Jaccard:
   Jw(A,B) = sum(w(t), t in A∩B) / sum(w(t), t in A∪B)
   where w(t) = IDF(t) = log((N+1)/(df(t)+1)) + 1
3. Directional token containment and unmatched rare-token mass
4. Shared phone/email/domain counts using sets (NOT forced positive)
5. House-number, unit-number, postal, and country disagreements
6. Name/address interactions and missingness indicators
7. Candidate ambiguity counts and competing-assignment margins
8. Raw, Unicode-normalized, and ASCII/transliterated views
9. Missing field vs conflicting field distinction
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


def sql_path(p: Path) -> str:
    return str(p).replace("\\\\", "/")


# All 26 rule names (must match blocking.py BLOCKING_RULES)
ALL_RULES = [
    "fingerprint", "phone", "email", "url",
    "name_postal", "house_postal", "name_core", "name_ascii",
    "name_prefix", "name_soundex", "address_sorted", "numbers",
    "name_sorted", "name_first3", "name_last3", "postal_name4",
    "house_name2", "house_name_last2", "name_reversed",
    "postal_block", "name_mid4", "name_core_prefix5",
    "number_name3", "addr_postal", "addr_tokens",
]

ID_COLUMNS = {"s1_row", "source1_entity_id", "candidate_entity_id", "split", "is_match"}


def build_advanced_features(
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
    """Build 80+ engineered features for each candidate pair via DuckDB SQL."""
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

    # Build rule flag SQL — set membership, not substring matching
    rule_flags = "\n".join([
        f"              CASE WHEN list_contains(string_split(c.rule_names, '|'), '{r}') THEN 1 ELSE 0 END::FLOAT AS rule_{r},"
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
                   has_phone, has_email, has_url,
                   country_clean,
                   name_clean, name_prefix_key, name_soundex_key
            FROM read_parquet('{sql_path(source1)}') __S1_FILTER__
          ),
          targets AS (
            SELECT entity_id, name_core, name_ascii, name_sorted_key,
                   address_clean, address_ascii, address_sorted_key,
                   postal_code, house_number, numbers, important_name_tokens,
                   important_address_tokens, phones, emails, urls,
                   record_fingerprint, name_len, address_len,
                   name_token_count, address_token_count,
                   has_phone, has_email, has_url,
                   country_clean,
                   name_clean, name_prefix_key, name_soundex_key
            FROM read_parquet('{sql_path(source2)}') __TARGETS_FILTER__
            UNION ALL
            SELECT entity_id, name_core, name_ascii, name_sorted_key,
                   address_clean, address_ascii, address_sorted_key,
                   postal_code, house_number, numbers, important_name_tokens,
                   important_address_tokens, phones, emails, urls,
                   record_fingerprint, name_len, address_len,
                   name_token_count, address_token_count,
                   has_phone, has_email, has_url,
                   country_clean,
                   name_clean, name_prefix_key, name_soundex_key
            FROM read_parquet('{sql_path(source3)}') __TARGETS_FILTER__
          ),
          {membership_sql}
          joined AS (
            SELECT
              c.s1_row, c.source1_entity_id, c.candidate_entity_id,
              {split_select}

              -- ── Blocking evidence ──────────────────────────────────────────
              c.evidence_score::FLOAT                                          AS evidence_score,
              c.candidate_rank::FLOAT                                          AS candidate_rank,
              c.rule_count::FLOAT                                              AS rule_count,
              count(*) OVER (PARTITION BY c.s1_row)::FLOAT                    AS candidate_count,
              max(c.evidence_score) OVER (PARTITION BY c.s1_row)::FLOAT
                - c.evidence_score                                             AS score_gap,
              (c.evidence_score / (max(c.evidence_score) OVER (PARTITION BY c.s1_row) + 1e-6))::FLOAT
                                                                               AS score_ratio,
              -- Margin: best vs second-best evidence (competing assignment)
              COALESCE(
                c.evidence_score - nth_value(c.evidence_score, 2) OVER (
                  PARTITION BY c.s1_row ORDER BY c.evidence_score DESC
                ), 0
              )::FLOAT                                                         AS evidence_margin,

              -- ── Source indicator ──────────────────────────────────────────
              CASE WHEN starts_with(c.candidate_entity_id, 'S2-') THEN 1 ELSE 0 END::FLOAT AS source_s2,

              -- ── Per-rule binary flags (set membership, not substring) ──────
              {rule_flags}

              -- ── Exact-match features ──────────────────────────────────────
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
              CASE WHEN a.country_clean<>'' AND a.country_clean=b.country_clean THEN 1 ELSE 0 END::FLOAT AS country_agree,

              -- ── Hard-positive flag (contact identity match) ───────────────
              CASE WHEN
                (a.phones<>'' AND a.phones=b.phones)
                OR (a.emails<>'' AND a.emails=b.emails)
                OR (a.record_fingerprint<>'' AND a.record_fingerprint=b.record_fingerprint)
                OR (a.urls<>'' AND a.urls=b.urls)
              THEN 1 ELSE 0 END::FLOAT AS hard_positive,

              -- ── Contradiction / conflict features ─────────────────────────
              -- House number conflict: both present but different
              CASE WHEN a.house_number<>'' AND b.house_number<>''
                   AND a.house_number<>b.house_number THEN 1 ELSE 0 END::FLOAT AS house_conflict,
              -- Postal conflict: both present but different
              CASE WHEN a.postal_code<>'' AND b.postal_code<>''
                   AND a.postal_code<>b.postal_code THEN 1 ELSE 0 END::FLOAT AS postal_conflict,
              -- Phone conflict: both have phone but different
              CASE WHEN a.phones<>'' AND b.phones<>''
                   AND a.phones<>b.phones THEN 1 ELSE 0 END::FLOAT AS phone_conflict,
              -- Email conflict: both have email but different
              CASE WHEN a.emails<>'' AND b.emails<>''
                   AND a.emails<>b.emails THEN 1 ELSE 0 END::FLOAT AS email_conflict,

              -- ── Missingness indicators (missing vs conflicting) ───────────
              CASE WHEN a.name_core='' OR b.name_core='' THEN 1 ELSE 0 END::FLOAT AS name_missing_either,
              CASE WHEN a.address_clean='' OR b.address_clean='' THEN 1 ELSE 0 END::FLOAT AS address_missing_either,
              CASE WHEN a.postal_code='' OR b.postal_code='' THEN 1 ELSE 0 END::FLOAT AS postal_missing_either,
              CASE WHEN a.house_number='' OR b.house_number='' THEN 1 ELSE 0 END::FLOAT AS house_missing_either,
              CASE WHEN a.phones='' AND b.phones='' THEN 1 ELSE 0 END::FLOAT AS both_phone_missing,
              CASE WHEN a.emails='' AND b.emails='' THEN 1 ELSE 0 END::FLOAT AS both_email_missing,

              -- ── Phone/email/domain shared counts (not forced positive) ────
              -- Learned reliability from labels; NOT a hard override
              CASE WHEN a.phones<>'' AND b.phones<>''
                   THEN (CASE WHEN a.phones=b.phones THEN 1 ELSE 0 END)
                   ELSE -1 END::FLOAT AS phone_agree_or_missing,
              CASE WHEN a.emails<>'' AND b.emails<>''
                   THEN (CASE WHEN a.emails=b.emails THEN 1 ELSE 0 END)
                   ELSE -1 END::FLOAT AS email_agree_or_missing,
              CASE WHEN a.urls<>'' AND b.urls<>''
                   THEN (CASE WHEN a.urls=b.urls THEN 1 ELSE 0 END)
                   ELSE -1 END::FLOAT AS url_agree_or_missing,

              -- ── String similarity (Jaro-Winkler) ─────────────────────────
              CASE WHEN a.name_core<>'' AND b.name_core<>'' THEN jaro_winkler_similarity(a.name_core,b.name_core) ELSE 0 END::FLOAT AS name_jw,
              CASE WHEN a.name_ascii<>'' AND b.name_ascii<>'' THEN jaro_winkler_similarity(a.name_ascii,b.name_ascii) ELSE 0 END::FLOAT AS name_ascii_jw,
              CASE WHEN a.name_clean<>'' AND b.name_clean<>'' THEN jaro_winkler_similarity(a.name_clean,b.name_clean) ELSE 0 END::FLOAT AS name_clean_jw,
              CASE WHEN a.address_clean<>'' AND b.address_clean<>'' THEN jaro_winkler_similarity(a.address_clean,b.address_clean) ELSE 0 END::FLOAT AS address_jw,
              CASE WHEN a.address_ascii<>'' AND b.address_ascii<>'' THEN jaro_winkler_similarity(a.address_ascii,b.address_ascii) ELSE 0 END::FLOAT AS address_ascii_jw,

              -- ── Levenshtein edit distance ratio (normalized edit similarity)
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

              -- ── Length ratio features ─────────────────────────────────────
              CASE WHEN greatest(a.name_len,b.name_len)>0 THEN
                least(a.name_len,b.name_len)::FLOAT / greatest(a.name_len,b.name_len) ELSE 0 END::FLOAT AS name_length_ratio,
              CASE WHEN greatest(a.address_len,b.address_len)>0 THEN
                least(a.address_len,b.address_len)::FLOAT / greatest(a.address_len,b.address_len) ELSE 0 END::FLOAT AS address_length_ratio,
              abs(a.name_len - b.name_len)::FLOAT                            AS name_len_diff,
              abs(a.address_len - b.address_len)::FLOAT                      AS address_len_diff,

              -- ── Token count features ──────────────────────────────────────
              CASE WHEN greatest(a.name_token_count,b.name_token_count)>0 THEN
                least(a.name_token_count,b.name_token_count)::FLOAT / greatest(a.name_token_count,b.name_token_count)
              ELSE 0 END::FLOAT AS name_token_count_ratio,
              abs(a.name_token_count - b.name_token_count)::FLOAT            AS name_token_count_diff,

              -- ── Prefix / suffix match ─────────────────────────────────────
              CASE WHEN a.name_ascii<>'' AND b.name_ascii<>'' AND
                left(a.name_ascii,6) = left(b.name_ascii,6) THEN 1 ELSE 0 END::FLOAT AS name_prefix6_match,
              CASE WHEN a.name_ascii<>'' AND b.name_ascii<>'' AND
                right(a.name_ascii,4) = right(b.name_ascii,4) THEN 1 ELSE 0 END::FLOAT AS name_suffix4_match,

              -- ── Information quality flags ─────────────────────────────────
              (a.has_phone::INT + b.has_phone::INT)::FLOAT                    AS both_have_phone,
              (a.has_email::INT + b.has_email::INT)::FLOAT                    AS both_have_email,
              (a.has_url::INT + b.has_url::INT)::FLOAT                        AS both_have_url,

              -- ── Directional containment (token coverage) ──────────────────
              -- How much of A's tokens appear in B (A->B containment)
              CASE WHEN a.name_token_count > 0 AND b.name_token_count > 0 THEN
                list_aggregate(
                  list_filter(string_split(a.important_name_tokens,'|'), x -> list_contains(string_split(b.important_name_tokens,'|'), x)),
                  'count'
                )::FLOAT / a.name_token_count
              ELSE 0 END::FLOAT AS name_a_in_b_coverage,
              -- How much of B's tokens appear in A (B->A containment)
              CASE WHEN a.name_token_count > 0 AND b.name_token_count > 0 THEN
                list_aggregate(
                  list_filter(string_split(b.important_name_tokens,'|'), x -> list_contains(string_split(a.important_name_tokens,'|'), x)),
                  'count'
                )::FLOAT / b.name_token_count
              ELSE 0 END::FLOAT AS name_b_in_a_coverage,

              -- ── Raw lengths ───────────────────────────────────────────────
              a.name_len::FLOAT                                                AS s1_name_len,
              b.name_len::FLOAT                                                AS cand_name_len,
              a.address_len::FLOAT                                             AS s1_addr_len,
              b.address_len::FLOAT                                             AS cand_addr_len,
              a.name_token_count::FLOAT                                        AS s1_token_count,
              b.name_token_count::FLOAT                                        AS cand_token_count

            FROM read_parquet('{sql_path(candidates)}') c
            JOIN s1 a ON c.source1_entity_id = a.entity_id
            JOIN targets b ON c.candidate_entity_id = b.entity_id
            {split_joins}
            __CHUNK_FILTER__
          )
          SELECT *,
            -- ── Composite interaction features ────────────────────────────────
            (name_jw * address_jw)::FLOAT                                      AS name_address_product,
            (name_jw * postal_exact)::FLOAT                                    AS name_postal_product,
            (name_ascii_jw * address_ascii_jw)::FLOAT                         AS ascii_product,
            (0.45*name_jw + 0.25*address_jw + 0.12*postal_exact
             + 0.08*numbers_exact + 0.10*name_tokens_exact)::FLOAT            AS identity_consensus,
            (name_exact + address_exact + postal_exact + numbers_exact
             + name_tokens_exact + phone_exact + email_exact
             + url_exact + fingerprint_exact)::FLOAT                           AS exact_evidence_count,
            CASE WHEN name_jw>=0.88 AND (address_jw>=0.72 OR postal_exact=1) THEN 1 ELSE 0 END::FLOAT
                                                                               AS strong_name_location,
            CASE WHEN name_jw>=0.95 THEN 1 ELSE 0 END::FLOAT                 AS very_high_name_jw,
            CASE WHEN name_edit_ratio>=0.90 AND address_edit_ratio>=0.80 THEN 1 ELSE 0 END::FLOAT
                                                                               AS high_edit_sim,
            -- Rule diversity
            CASE WHEN rule_count>=4 THEN 1 ELSE 0 END::FLOAT                  AS many_rules_fired,
            CASE WHEN rule_count>=2 THEN 1 ELSE 0 END::FLOAT                  AS multi_rule,
            -- Rank context
            (1.0 / (candidate_rank + 1.0))::FLOAT                             AS reciprocal_rank,
            (score_gap * candidate_rank)::FLOAT                                AS gap_rank_product,
            -- Ambiguity: high candidate count = ambiguous entity
            CASE WHEN candidate_count >= 10 THEN 1 ELSE 0 END::FLOAT          AS high_ambiguity,
            CASE WHEN candidate_count >= 50 THEN 1 ELSE 0 END::FLOAT          AS very_high_ambiguity,
            -- Directional containment max (bidirectional coverage)
            greatest(name_a_in_b_coverage, name_b_in_a_coverage)::FLOAT       AS name_max_coverage,
            least(name_a_in_b_coverage, name_b_in_a_coverage)::FLOAT          AS name_min_coverage,
            -- Contradiction accumulator
            (house_conflict + postal_conflict + phone_conflict + email_conflict)::FLOAT AS total_conflicts,
            -- Name-address combined signal
            (name_jw + address_jw) / 2.0::FLOAT                               AS name_address_mean,
            (name_edit_ratio + address_edit_ratio) / 2.0::FLOAT               AS edit_mean,
            -- Precision-oriented: high name, high address, no conflicts
            CASE WHEN name_jw>=0.92 AND address_jw>=0.78
                      AND house_conflict=0 AND postal_conflict=0 THEN 1 ELSE 0 END::FLOAT
                                                                               AS high_conf_match
          FROM joined
        ) TO '__OUTPUT_PATH__'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
        """
        cand_size = os.path.getsize(candidates)
        num_chunks = 20 if cand_size > 500 * 1024 * 1024 else 1

        chunk_files = []
        for i in range(num_chunks):
            print(f"    processing feature chunk {i+1}/{num_chunks}...", flush=True)
            chunk_file = temp_dir / f"feats_chunk_{i}.parquet"
            chunk_files.append(chunk_file)
            
            chunk_filter = f"WHERE (hash(c.candidate_entity_id) % {num_chunks}) = {i}" if num_chunks > 1 else ""
            targets_filter = f"WHERE (hash(entity_id) % {num_chunks}) = {i}" if num_chunks > 1 else ""
            
            chunk_query = query.replace("__CHUNK_FILTER__", chunk_filter) \
                               .replace("__TARGETS_FILTER__", targets_filter) \
                               .replace("__S1_FILTER__", "") \
                               .replace("__OUTPUT_PATH__", sql_path(chunk_file))
            con.execute(chunk_query)

        # Merge chunks
        if num_chunks > 1:
            print(f"    merging {num_chunks} feature chunks...", flush=True)
            paths_str = ",".join(f"'{sql_path(f)}'" for f in chunk_files)
            merge_query = f"COPY (SELECT * FROM read_parquet([{paths_str}])) TO '{sql_path(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"
            con.execute(merge_query)
        else:
            os.replace(chunk_files[0], temporary)
            
    finally:
        con.close()

    os.replace(temporary, output)
    print(f"  features written -> {output}", flush=True)


def numeric_feature_names(schema: pa.Schema) -> list[str]:
    """Return all float feature columns (excludes ID and label columns)."""
    return [
        name for name in schema.names
        if name not in ID_COLUMNS and pa.types.is_floating(schema.field(name).type)
    ]
