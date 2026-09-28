"""
Advanced Hybrid Pipeline — End-to-End Orchestrator.

Orchestrates all 7 phases:
  Phase 0: Prepare filtered Source-1 for model training (validation + hash sample)
  Phase 1: Ultra-high-recall candidate blocking (25 rules, hash-partitioned aggregation)
  Phase 2: Rich feature engineering (80+ features, agreement/contradiction/margin)
  Phase 3: LightGBM ensemble training & macro F0.5 threshold optimization
  Phase 4: Test candidate blocking
  Phase 5: Test feature engineering
  Phase 6: Test scoring
  Phase 7: Format, deduplicate, validate, and write final submission & evaluation report

Usage:
  python -m src.advanced_hybrid.pipeline [options]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import lightgbm as lgb
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.advanced_hybrid.blocking import (
    build_advanced_hybrid_candidates,
    measure_candidate_recall,
    validate_candidate_schema,
    sql_path,
)
from src.advanced_hybrid.features import (
    build_advanced_features,
    numeric_feature_names,
)
from src.advanced_hybrid.model import (
    train_advanced_lgbm,
    score_features_file,
    write_submission_files,
    compute_macro_f05,
    compute_bootstrap_ci,
    truth_counts,
)


def prepare_training_source1(
    full_source1: Path,
    train_truth: Path,
    valid_truth: Path,
    output: Path,
    sample_modulus: int = 5,
    force: bool = False,
) -> None:
    """Keep all validation entities plus a deterministic hash sample of train entities."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if not force and output.exists():
        print(f"  training source1 [cached] -> {output}", flush=True)
        return

    temporary = output.with_suffix(".tmp.parquet")
    con = duckdb.connect()
    try:
        con.execute(f"""
        COPY (
          WITH valid_ids AS (
            SELECT source1_entity_id
            FROM read_csv('{sql_path(valid_truth)}', delim='\\t', header=true, all_varchar=true)
          ),
          train_ids AS (
            SELECT source1_entity_id
            FROM read_csv('{sql_path(train_truth)}', delim='\\t', header=true, all_varchar=true)
            WHERE abs(hash(source1_entity_id)) % {max(1, sample_modulus)} = 0
          ),
          selected AS (
            SELECT source1_entity_id FROM valid_ids
            UNION ALL
            SELECT source1_entity_id FROM train_ids
          )
          SELECT s.*
          FROM read_parquet('{sql_path(full_source1)}') s
          JOIN selected x ON s.entity_id = x.source1_entity_id
        ) TO '{sql_path(temporary)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
        """)
    finally:
        con.close()
    os.replace(temporary, output)
    print(f"  training source1 written -> {output}", flush=True)


def validate_submission_integrity(
    matching_tsv: Path,
    candidate_tsv: Path,
    source1_clean: Path,
) -> dict[str, Any]:
    """Strict verification of submission integrity per specification."""
    # 1. Count rows in source1
    pf = pq.ParquetFile(source1_clean)
    expected_rows = pf.metadata.num_rows

    # 2. Check headers and line counts
    with matching_tsv.open("r", encoding="utf-8") as f:
        match_header = f.readline().strip()
        match_lines = sum(1 for _ in f)

    with candidate_tsv.open("r", encoding="utf-8") as f:
        cand_header = f.readline().strip()
        cand_lines = sum(1 for _ in f)

    assert match_header == "source1_entity_id\tmatched_entity_ids", f"Invalid match header: {match_header}"
    assert cand_header == "source1_entity_id\tcandidate_entity_ids", f"Invalid cand header: {cand_header}"
    assert match_lines == expected_rows, f"Match line count {match_lines} != expected {expected_rows}"
    assert cand_lines == expected_rows, f"Cand line count {cand_lines} != expected {expected_rows}"

    # 3. Check every match is in candidates and no duplicates
    violations = 0
    duplicate_violations = 0
    total_matches = 0
    singletons = 0

    with (
        matching_tsv.open("r", encoding="utf-8") as mf,
        candidate_tsv.open("r", encoding="utf-8") as cf,
    ):
        next(mf)
        next(cf)
        for m_line, c_line in zip(mf, cf):
            m_parts = m_line.strip().split("\t")
            c_parts = c_line.strip().split("\t")

            s1_m = m_parts[0]
            s1_c = c_parts[0]
            assert s1_m == s1_c, f"ID alignment error: {s1_m} vs {s1_c}"

            m_ids = [x for x in m_parts[1].split(",") if x] if len(m_parts) > 1 else []
            c_ids = [x for x in c_parts[1].split(",") if x] if len(c_parts) > 1 else []

            if len(m_ids) != len(set(m_ids)):
                duplicate_violations += 1

            c_set = set(c_ids)
            for m in m_ids:
                if m not in c_set:
                    violations += 1

            if not m_ids:
                singletons += 1
            total_matches += len(m_ids)

    return {
        "status": "PASSED" if violations == 0 and duplicate_violations == 0 else "FAILED",
        "expected_entities": expected_rows,
        "match_rows": match_lines,
        "candidate_rows": cand_lines,
        "total_matches": total_matches,
        "singleton_entities": singletons,
        "matches_missing_from_candidates": violations,
        "duplicate_match_violations": duplicate_violations,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Advanced Hybrid Pipeline Runner")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--work-dir", default="data/advanced_hybrid")
    parser.add_argument("--output-dir", default="output/advanced_hybrid")
    parser.add_argument("--model-dir", default="models/advanced_hybrid")
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 4)))
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--sample-modulus", type=int, default=6)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    return parser.parse_args()


def run_pipeline(args=None):
    if args is None:
        args = parse_args()

    start_time = time.perf_counter()
    data_dir = Path(args.data_dir)
    split_dir = Path(args.split_dir)
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)

    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Input paths
    train_s1 = data_dir / "train_source1_clean.parquet"
    train_s2 = data_dir / "train_source2_clean.parquet"
    train_s3 = data_dir / "train_source3_clean.parquet"
    test_s1 = data_dir / "test_source1_clean.parquet"
    test_s2 = data_dir / "test_source2_clean.parquet"
    test_s3 = data_dir / "test_source3_clean.parquet"

    train_truth = split_dir / "train_ground_truth.tsv"
    valid_truth = split_dir / "valid_ground_truth.tsv"

    # Artifact paths
    model_s1 = work_dir / "model_train_source1.parquet"
    train_cands = work_dir / "train_candidates.parquet"
    train_feats = work_dir / "train_features.parquet"
    test_cands = work_dir / "test_candidates.parquet"
    test_feats = work_dir / "test_features.parquet"
    test_preds = work_dir / "test_predictions.parquet"

    final_matching = output_dir / "matching_results.tsv"
    final_candidates = output_dir / "candidate_pairs.tsv"
    final_report = output_dir / "evaluation_report.json"

    print("=" * 70, flush=True)
    print(" ADVANCED HYBRID ENTITY RESOLUTION PIPELINE", flush=True)
    print(" Target: Macro F0.5 Optimization with Complete Correctness", flush=True)
    print("=" * 70, flush=True)

    # ── Phase 0: Prepare filtered training source1 ─────────────────────────
    print("\nPhase 0/7: Selecting validation IDs + deterministic training sample...", flush=True)
    prepare_training_source1(
        full_source1=train_s1,
        train_truth=train_truth,
        valid_truth=valid_truth,
        output=model_s1,
        sample_modulus=args.sample_modulus,
        force=args.force,
    )

    # ── Phase 1: High-recall candidate blocking (Train) ───────────────────
    print("\nPhase 1/7: Generating ultra-high-recall training candidates...", flush=True)
    build_advanced_hybrid_candidates(
        source1=model_s1,
        source2=train_s2,
        source3=train_s3,
        output=train_cands,
        temp_dir=work_dir / "tmp_train_cands",
        threads=args.threads,
        memory_limit=args.memory_limit,
        max_candidates=args.max_candidates,
        force=args.force,
    )

    print("  measuring validation recall...", flush=True)
    blocking_recall = measure_candidate_recall(
        candidates=train_cands,
        ground_truth=valid_truth,
        output_json=work_dir / "validation_blocking_recall.json",
    )

    # ── Phase 2: Feature engineering (Train) ──────────────────────────────
    print("\nPhase 2/7: Building 80+ agreement/contradiction features (Train)...", flush=True)
    if args.force or not train_feats.exists():
        build_advanced_features(
            candidates=train_cands,
            source1=model_s1,
            source2=train_s2,
            source3=train_s3,
            output=train_feats,
            temp_dir=work_dir / "tmp_train_feats",
            threads=args.threads,
            memory_limit=args.memory_limit,
            train_truth=train_truth,
            valid_truth=valid_truth,
        )
    else:
        print(f"  training features [cached] -> {train_feats}", flush=True)

    # ── Phase 3: Train model & calibrate threshold ────────────────────────
    print("\nPhase 3/7: Training LightGBM ensemble & calibrating macro F0.5...", flush=True)
    pf_feats = pq.ParquetFile(train_feats)
    feat_cols = numeric_feature_names(pf_feats.schema_arrow)
    print(f"  feature count: {len(feat_cols)}", flush=True)

    # Use DuckDB to efficiently filter and sample without loading 55M rows into memory
    needed_cols = feat_cols + ["split", "is_match", "source1_entity_id", "candidate_entity_id"]
    cols_sql = ", ".join(needed_cols)
    max_train_pairs = 1_200_000

    con = duckdb.connect()
    try:
        con.execute(f"SET threads = {max(1, args.threads)}")
        con.execute(f"SET memory_limit = '{args.memory_limit}'")
        con.execute(f"SET temp_directory = '{sql_path(work_dir / 'tmp_train_feats')}'")

        print("  loading validation features...", flush=True)
        valid_df = con.execute(f"""
            SELECT {cols_sql}
            FROM read_parquet('{sql_path(train_feats)}')
            WHERE split = 'valid'
        """).df()
        
        print("  loading and subsampling train features...", flush=True)
        # First, find out how many positives we have
        pos_count = con.execute(f"""
            SELECT count(*) 
            FROM read_parquet('{sql_path(train_feats)}')
            WHERE split = 'train' AND is_match = 1
        """).fetchone()[0]
        
        sample_n = max(10, max_train_pairs - pos_count)
        
        train_df = con.execute(f"""
            WITH train_all AS (
                SELECT {cols_sql}
                FROM read_parquet('{sql_path(train_feats)}')
                WHERE split = 'train'
            ),
            positives AS (
                SELECT * FROM train_all WHERE is_match = 1
            ),
            negatives AS (
                SELECT * FROM train_all WHERE is_match = 0
            ),
            sampled_negatives AS (
                SELECT * FROM negatives
                USING SAMPLE reservoir({sample_n} ROWS)
                REPEATABLE ({args.seed})
            )
            SELECT * FROM positives
            UNION ALL
            SELECT * FROM sampled_negatives
        """).df()
    finally:
        con.close()

    # Convert to float32 to reduce memory footprint and shuffle train
    train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    for col in feat_cols:
        train_df[col] = train_df[col].astype(np.float32)
        valid_df[col] = valid_df[col].astype(np.float32)
        
    pos_c = int(train_df["is_match"].sum())
    print(f"  subsampled training pairs to {len(train_df):,} ({pos_c:,} pos, {len(train_df) - pos_c:,} neg)", flush=True)

    print(f"  training pairs: {len(train_df):,} | validation pairs: {len(valid_df):,}", flush=True)
    model_metadata, valid_metrics = train_advanced_lgbm(
        train_df=train_df,
        valid_df=valid_df,
        feat_names=feat_cols,
        valid_truth=valid_truth,
        model_dir=model_dir,
        n_seeds=args.n_seeds,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    # Compute bootstrap CI on validation set
    print("  computing bootstrap confidence intervals...", flush=True)
    truth_map, _ = truth_counts(valid_truth)
    # Re-evaluate best threshold predictions
    best_thresh = model_metadata["threshold"]
    v_cands = valid_df["candidate_entity_id"].to_numpy()
    v_s1 = valid_df["source1_entity_id"].to_numpy()
    # Score valid with trained ensemble
    boosters = [
        lgb.Booster(model_str=Path(entry["booster_path"]).read_text(encoding="utf-8"))
        for entry in model_metadata["boosters"]
    ]
    x_val = valid_df[feat_cols].to_numpy(dtype=np.float32)
    val_scores = np.zeros(len(valid_df), dtype=np.float64)
    for b in boosters:
        val_scores += b.predict(x_val) / len(boosters)

    pred_map = {}
    for s1, c, sc in zip(v_s1, v_cands, val_scores):
        if sc >= best_thresh:
            pred_map.setdefault(str(s1), []).append(str(c))

    bootstrap_ci = compute_bootstrap_ci(pred_map, truth_map, n_bootstrap=400, seed=args.seed)
    valid_metrics["bootstrap_f0_5"] = bootstrap_ci

    print(f"  Measured Validation Macro F0.5: {valid_metrics['macro_f0_5']:.5f}", flush=True)
    print(f"  95% CI: [{bootstrap_ci['ci_lower_95']:.5f}, {bootstrap_ci['ci_upper_95']:.5f}]", flush=True)

    if args.validation_only:
        eval_report = {
            "pipeline": "advanced_hybrid",
            "mode": "validation_only",
            "blocking_recall": blocking_recall,
            "validation_metrics": valid_metrics,
            "elapsed_minutes": round((time.perf_counter() - start_time) / 60, 2),
        }
        final_report.write_text(json.dumps(eval_report, indent=2), encoding="utf-8")
        print(f"\nValidation report saved -> {final_report}", flush=True)
        return

    # ── Phase 4: Test candidate blocking ──────────────────────────────────
    print("\nPhase 4/7: Generating test candidates (25 blocking rules)...", flush=True)
    build_advanced_hybrid_candidates(
        source1=test_s1,
        source2=test_s2,
        source3=test_s3,
        output=test_cands,
        temp_dir=work_dir / "tmp_test_cands",
        threads=args.threads,
        memory_limit=args.memory_limit,
        max_candidates=args.max_candidates,
        force=args.force,
    )

    # ── Phase 5: Test feature engineering ─────────────────────────────────
    print("\nPhase 5/7: Building test features...", flush=True)
    if args.force or not test_feats.exists():
        build_advanced_features(
            candidates=test_cands,
            source1=test_s1,
            source2=test_s2,
            source3=test_s3,
            output=test_feats,
            temp_dir=work_dir / "tmp_test_feats",
            threads=args.threads,
            memory_limit=args.memory_limit,
        )
    else:
        print(f"  test features [cached] -> {test_feats}", flush=True)

    # ── Phase 6: Score test pairs ─────────────────────────────────────────
    print("\nPhase 6/7: Scoring test pairs with ensemble...", flush=True)
    score_features_file(
        features_parquet=test_feats,
        output_predictions_parquet=test_preds,
        model_metadata=model_metadata,
    )

    # ── Phase 7: Format, deduplicate, validate submission ─────────────────
    print("\nPhase 7/7: Formatting submission and verifying integrity...", flush=True)
    sub_stats = write_submission_files(
        source1_clean_parquet=test_s1,
        predictions_parquet=test_preds,
        matching_tsv_path=final_matching,
        candidate_tsv_path=final_candidates,
        threshold=best_thresh,
        max_matches=11,
    )

    integrity_check = validate_submission_integrity(
        matching_tsv=final_matching,
        candidate_tsv=final_candidates,
        source1_clean=test_s1,
    )

    elapsed_min = round((time.perf_counter() - start_time) / 60, 2)

    eval_report = {
        "pipeline": "advanced_hybrid",
        "status": "COMPLETED",
        "elapsed_minutes": elapsed_min,
        "target_goal": "Macro F0.5 >= 0.999 (measured evidence)",
        "measured_validation_macro_f0_5": valid_metrics["macro_f0_5"],
        "measured_validation_macro_f1": valid_metrics["macro_f1"],
        "measured_pair_precision": valid_metrics["pair_precision"],
        "measured_pair_recall": valid_metrics["pair_recall"],
        "singleton_fpr": valid_metrics["singleton_fpr"],
        "bootstrap_95_ci": bootstrap_ci,
        "blocking_recall": blocking_recall,
        "validation_metrics": valid_metrics,
        "submission_statistics": sub_stats,
        "integrity_verification": integrity_check,
        "artifacts": {
            "matching_results": str(final_matching),
            "candidate_pairs": str(final_candidates),
            "evaluation_report": str(final_report),
            "model_dir": str(model_dir),
        },
    }

    final_report.write_text(json.dumps(eval_report, indent=2), encoding="utf-8")
    print("\n" + "=" * 70, flush=True)
    print(" PIPELINE EXECUTION COMPLETE", flush=True)
    print("=" * 70, flush=True)
    print(json.dumps(eval_report, indent=2), flush=True)


if __name__ == "__main__":
    run_pipeline()
