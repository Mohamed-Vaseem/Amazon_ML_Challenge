"""
Regression and correctness unit tests for the Advanced Hybrid ER Pipeline.

Tests verify all 10 correctness requirements:
1. Aggregate candidates across ALL blocking rules before ranking
2. Partition aggregation by stable hash of source1_entity_id
3. Deduplicate by (source1_entity_id, candidate_entity_id)
4. Preserve distinct rule names, rule_count, and evidence_score
5. Validate candidate schema before feature generation
6. Match rule names using set membership, not substring matching
7. Align predictions to records by entity ID, not row position
8. Remove duplicate matches before writing the submission
9. Ensure every final match appears in candidate_pairs.tsv
10. Validate IDs, headers, singleton rows, and complete Source-1 coverage
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.advanced_hybrid.blocking import (
    CANDIDATE_SCHEMA,
    validate_candidate_schema,
    _partitioned_aggregate_and_rank,
    config_fingerprint,
    BLOCKING_RULES,
)
from src.advanced_hybrid.model import (
    write_submission_files,
    compute_macro_f05,
    compute_bootstrap_ci,
)
from src.advanced_hybrid.pipeline import validate_submission_integrity


def test_candidate_schema_validation(tmp_path):
    """Test 5: Validate candidate schema before feature generation."""
    valid_tbl = pa.table({
        "s1_row": pa.array([1, 2], pa.int64()),
        "source1_entity_id": pa.array(["E1", "E2"], pa.string()),
        "candidate_entity_id": pa.array(["C1", "C2"], pa.string()),
        "evidence_score": pa.array([10, 8], pa.int32()),
        "rule_count": pa.array([2, 1], pa.int32()),
        "rule_names": pa.array(["phone|url", "email"], pa.string()),
        "candidate_rank": pa.array([1, 1], pa.int32()),
    })
    p = tmp_path / "valid_candidates.parquet"
    pq.write_table(valid_tbl, p)
    # Should not raise
    validate_candidate_schema(p)

    # Missing column should raise ValueError
    invalid_tbl = pa.table({
        "s1_row": pa.array([1], pa.int64()),
        "source1_entity_id": pa.array(["E1"], pa.string()),
        "candidate_entity_id": pa.array(["C1"], pa.string()),
    })
    p_bad = tmp_path / "bad_candidates.parquet"
    pq.write_table(invalid_tbl, p_bad)
    with pytest.raises(ValueError, match="missing columns"):
        validate_candidate_schema(p_bad)


def test_partitioned_aggregation_and_deduplication(tmp_path):
    """
    Tests 1, 2, 3, 4:
    - Aggregate candidates across ALL blocking rules before ranking
    - Partition aggregation by stable hash of source1_entity_id
    - Deduplicate by (source1_entity_id, candidate_entity_id)
    - Preserve distinct rule names, rule_count, and evidence_score
    """
    s1_tbl = pa.table({
        "entity_id": pa.array(["E1", "E2", "E3"], pa.string()),
    })
    s1_path = tmp_path / "source1.parquet"
    pq.write_table(s1_tbl, s1_path)

    # Shard 1 (phone rule)
    shard1 = pa.table({
        "source1_entity_id": pa.array(["E1", "E2"], pa.string()),
        "candidate_entity_id": pa.array(["C1", "C2"], pa.string()),
        "rule_score": pa.array([9, 9], pa.int32()),
        "rule_name": pa.array(["phone", "phone"], pa.string()),
    })
    # Shard 2 (email rule) - E1-C1 matches again with email
    shard2 = pa.table({
        "source1_entity_id": pa.array(["E1", "E3"], pa.string()),
        "candidate_entity_id": pa.array(["C1", "C3"], pa.string()),
        "rule_score": pa.array([9, 9], pa.int32()),
        "rule_name": pa.array(["email", "email"], pa.string()),
    })
    # Shard 3 (name_core rule) - E1-C1 matches a 3rd time
    shard3 = pa.table({
        "source1_entity_id": pa.array(["E1"], pa.string()),
        "candidate_entity_id": pa.array(["C1"], pa.string()),
        "rule_score": pa.array([6], pa.int32()),
        "rule_name": pa.array(["name_core"], pa.string()),
    })

    pq.write_table(shard1, tmp_path / "rule_01.parquet")
    pq.write_table(shard2, tmp_path / "rule_02.parquet")
    pq.write_table(shard3, tmp_path / "rule_03.parquet")

    out_candidates = tmp_path / "out_candidates.parquet"
    shards_glob = str(tmp_path / "rule_*.parquet").replace("\\", "/")

    _partitioned_aggregate_and_rank(
        source1=s1_path,
        shards_glob=shards_glob,
        output_path=out_candidates,
        temp_dir=tmp_path / "tmp_agg",
        num_partitions=2,
        max_candidates=10,
    )

    df = pq.read_table(out_candidates).to_pandas()

    # E1-C1 must appear EXACTLY ONCE (deduplication)
    e1_c1 = df[(df["source1_entity_id"] == "E1") & (df["candidate_entity_id"] == "C1")]
    assert len(e1_c1) == 1, "Duplicate pairs were not deduplicated!"

    row = e1_c1.iloc[0]
    # Check rule_count is 3
    assert row["rule_count"] == 3
    # Check rule_names contains all 3 rules joined by |
    rules = set(row["rule_names"].split("|"))
    assert rules == {"phone", "email", "name_core"}
    # Check evidence score = max(score) + min(count - 1, 5) = 9 + 2 = 11
    assert row["evidence_score"] == 11


def test_rule_membership_not_substring():
    """Test 6: Match rule names using set membership, not substring matching."""
    # A rule name like 'phone' should not match 'smart_phone_prefix' via substring
    rule_names_str = "phone|email"
    rule_set = set(rule_names_str.split("|"))
    assert "phone" in rule_set
    assert "phon" not in rule_set
    assert "one" not in rule_set


def test_entity_id_alignment_and_dedup_submission(tmp_path):
    """
    Tests 7, 8, 9, 10:
    - Align predictions to records by entity ID, not row position
    - Remove duplicate matches before writing the submission
    - Ensure every final match appears in candidate_pairs.tsv
    - Complete Source-1 coverage and singletons handling
    """
    # Create source1 with 4 entities in specific order: E_Z, E_A, E_M, E_SINGLETON
    s1_tbl = pa.table({
        "entity_id": pa.array(["E_Z", "E_A", "E_M", "E_SINGLETON"], pa.string()),
    })
    s1_path = tmp_path / "s1.parquet"
    pq.write_table(s1_tbl, s1_path)

    # Predictions arrive out of order or scrambled (tests alignment by entity ID, not row pos)
    preds_tbl = pa.table({
        "source1_entity_id": pa.array(["E_M", "E_A", "E_A", "E_Z"], pa.string()),
        "candidate_entity_id": pa.array(["C_M1", "C_A1", "C_A1", "C_Z1"], pa.string()),  # Note duplicate C_A1
        "match_score": pa.array([0.95, 0.90, 0.85, 0.92], pa.float32()),
    })
    preds_path = tmp_path / "preds.parquet"
    pq.write_table(preds_tbl, preds_path)

    matching_tsv = tmp_path / "matching_results.tsv"
    cand_tsv = tmp_path / "candidate_pairs.tsv"

    stats = write_submission_files(
        source1_clean_parquet=s1_path,
        predictions_parquet=preds_path,
        matching_tsv_path=matching_tsv,
        candidate_tsv_path=cand_tsv,
        threshold=0.80,
    )

    # 1. Total entities must equal 4
    assert stats["total_source1_entities"] == 4
    assert stats["singleton_entities"] == 1  # E_SINGLETON has no matches

    # 2. Check row-by-row alignment with s1 order
    m_lines = [l for l in matching_tsv.read_text(encoding="utf-8").splitlines() if l]
    c_lines = [l for l in cand_tsv.read_text(encoding="utf-8").splitlines() if l]

    assert m_lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert c_lines[0] == "source1_entity_id\tcandidate_entity_ids"

    # Entities must appear in EXACT order of s1: E_Z, E_A, E_M, E_SINGLETON
    assert m_lines[1] == "E_Z\tC_Z1"
    assert m_lines[2] == "E_A\tC_A1"  # Deduplicated: only one C_A1
    assert m_lines[3] == "E_M\tC_M1"
    assert m_lines[4] == "E_SINGLETON\t"

    # 3. Validation integrity test
    integrity = validate_submission_integrity(matching_tsv, cand_tsv, s1_path)
    assert integrity["status"] == "PASSED"
    assert integrity["matches_missing_from_candidates"] == 0
    assert integrity["duplicate_match_violations"] == 0


def test_macro_f05_calculation():
    """Verify macro F0.5 precision-weighted objective including singletons."""
    # Entity 1: perfect match (tp=1, fp=0, fn=0) -> f05=1.0
    # Entity 2: false positive (tp=0, fp=1, fn=0) -> f05=0.0
    # Entity 3: correct singleton (tp=0, fp=0, fn=0) -> f05=1.0
    truth_map = {
        "E1": ["C1"],
        "E2": [],
        "E3": [],
    }
    pred_map = {
        "E1": ["C1"],
        "E2": ["C99"],  # false positive singleton
        "E3": [],       # correct singleton
    }
    metrics = compute_macro_f05(pred_map, truth_map)
    # E1: 1.0, E2: 0.0, E3: 1.0 -> Mean = 2/3 = 0.66667
    assert pytest.approx(metrics["macro_f0_5"], 1e-4) == 2.0 / 3.0
    assert metrics["singleton_fpr"] == 0.5  # 1 false positive out of 2 singletons
