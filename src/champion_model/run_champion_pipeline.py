"""Champion Model Pipeline — End-to-End Runner.

Full pipeline:
  Phase 0: Prepare filtered source1 for training (deterministic sample + all valid IDs)
  Phase 1: Generate ultra-high-recall training candidates (24 blocking rules)
  Phase 2: Build rich features (60+ features) for training pairs
  Phase 3: Train LightGBM ensemble (or improved logistic fallback)
  Phase 4: Generate test candidates
  Phase 5: Build rich features for test pairs
  Phase 6: Score test pairs
  Phase 7: Write and validate final submission

Run:
  python -m src.champion_model.run_champion_pipeline

Key improvements over baseline:
  - Blocking recall target: ≥ 92% (vs 77.7% baseline)
  - LightGBM ensemble vs single logistic regression
  - 60+ features vs 47
  - Hard-positive overrides (fingerprint/phone/email/url exact)
  - macro_F0.5 threshold tuning (vs macro_F1)
  - 3 seeds × 5 folds = 15 models averaged
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.champion_model.blocking import build_ultra_recall_candidates, sql_path
from src.champion_model.features import build_rich_features, numeric_feature_names
from src.champion_model.model import (
    train_champion_model,
    load_and_score,
    HARD_POSITIVE_FEATURES,
)


# ─────────────────────────────────────────────────────────────────
#  Helper: select source1 rows for model training
# ─────────────────────────────────────────────────────────────────
def prepare_model_source1(
    full_source1: Path,
    train_truth: Path,
    valid_truth: Path,
    output: Path,
    sample_modulus: int = 6,   # more training rows than baseline (was 10)
) -> None:
    """Keep all validation entities plus a deterministic training sample."""
    output.parent.mkdir(parents=True, exist_ok=True)
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
            WHERE hash(source1_entity_id) % {max(1, sample_modulus)} = 0
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
    print(f"  model source1 written: {output}", flush=True)


# ─────────────────────────────────────────────────────────────────
#  Helper: iterate source1 entity IDs in row order
# ─────────────────────────────────────────────────────────────────
def iter_source1_ids(source1: Path):
    for batch in pq.ParquetFile(source1).iter_batches(
        batch_size=100_000, columns=["entity_id"]
    ):
        yield from batch.column("entity_id").to_pylist()


# ─────────────────────────────────────────────────────────────────
#  Helper: iterate prediction groups
# ─────────────────────────────────────────────────────────────────
def iter_prediction_groups(path: Path):
    current = None
    source_id = None
    values = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=250_000):
        frame = batch.to_pandas().sort_values(
            ["s1_row", "match_score"], ascending=[True, False]
        )
        for row in frame.itertuples(index=False):
            number = int(row.s1_row)
            if current is not None and number != current:
                yield current, source_id, values
                values = []
            current = number
            source_id = str(row.source1_entity_id)
            values.append((str(row.candidate_entity_id), float(row.match_score)))
    if current is not None:
        yield current, source_id, values


# ─────────────────────────────────────────────────────────────────
#  Helper: write final submission files
# ─────────────────────────────────────────────────────────────────
def write_submission(
    source1: Path,
    predictions: Path,
    matching: Path,
    candidate_tsv: Path,
    threshold: float,
    max_matches: int = 11,
) -> dict[str, object]:
    matching.parent.mkdir(parents=True, exist_ok=True)
    groups = iter(iter_prediction_groups(predictions))
    current = next(groups, None)

    total_s1 = 0
    total_matched = 0
    with (
        matching.open("w", encoding="utf-8", newline="") as mf,
        candidate_tsv.open("w", encoding="utf-8", newline="") as cf,
    ):
        mf.write("source1_entity_id\tmatched_entity_ids\n")
        cf.write("source1_entity_id\tcandidate_entity_ids\n")
        for row_number, source_id in enumerate(iter_source1_ids(source1), start=1):
            total_s1 += 1
            pairs = []
            if current is not None and current[0] == row_number:
                pairs = current[2]
                current = next(groups, None)
            candidate_ids = [v for v, _ in pairs]
            match_ids = [v for v, s in pairs if s >= threshold][:max_matches]
            total_matched += len(match_ids)
            mf.write(f"{source_id}\t{','.join(match_ids)}\n")
            cf.write(f"{source_id}\t{','.join(candidate_ids)}\n")

    return {
        "total_source1_rows": total_s1,
        "total_matched_pairs": total_matched,
        "avg_matches_per_entity": round(total_matched / max(1, total_s1), 4),
        "threshold_used": threshold,
    }


# ─────────────────────────────────────────────────────────────────
#  Argument parsing
# ─────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the champion entity-resolution pipeline."
    )
    parser.add_argument("--processed-dir",  default="data/processed")
    parser.add_argument("--split-dir",      default="data/splits")
    parser.add_argument("--work-dir",       default="data/champion")
    parser.add_argument("--output-dir",     default="output/champion")
    parser.add_argument("--model",          default="models/champion_model.json")
    parser.add_argument("--threads",        type=int,
                        default=max(1, min(8, os.cpu_count() or 4)))
    parser.add_argument("--memory-limit",   default="10GB")
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--max-matches",    type=int, default=11)
    parser.add_argument("--max-train-rows", type=int, default=2_000_000)
    parser.add_argument("--train-sample-modulus", type=int, default=6)
    parser.add_argument("--n-seeds",        type=int, default=3)
    parser.add_argument("--n-folds",        type=int, default=5)
    parser.add_argument("--seed",           type=int, default=2026)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--force",          action="store_true")
    # Allow skipping individual phases (for resuming after crash)
    parser.add_argument("--skip-train-candidates", action="store_true")
    parser.add_argument("--skip-train-features",   action="store_true")
    parser.add_argument("--skip-train-model",      action="store_true")
    parser.add_argument("--skip-test-candidates",  action="store_true")
    parser.add_argument("--skip-test-features",    action="store_true")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    started = time.perf_counter()

    processed = Path(args.processed_dir)
    splits    = Path(args.split_dir)
    work      = Path(args.work_dir)
    output    = Path(args.output_dir)
    model_path = Path(args.model)
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    temp = work / "duckdb_tmp"

    # ── File paths ─────────────────────────────────────────────
    train_sources = [processed / f"train_source{i}_clean.parquet" for i in (1, 2, 3)]
    test_sources  = [processed / f"test_source{i}_clean.parquet"  for i in (1, 2, 3)]
    train_truth   = splits / "train_ground_truth.tsv"
    valid_truth   = splits / "valid_ground_truth.tsv"

    required = train_sources + test_sources + [train_truth, valid_truth]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")

    model_source1       = work / "model_train_source1.parquet"
    train_candidates    = work / "train_candidates.parquet"
    train_features      = work / "train_features.parquet"
    test_candidates     = work / "test_candidates.parquet"
    test_features       = work / "test_features.parquet"
    predictions         = output / "test_predictions.parquet"
    metrics_path        = output / "validation_metrics.json"
    matching            = output / "matching_results.tsv"
    candidate_tsv_out   = output / "candidate_pairs.tsv"

    # ── Phase 0: Prepare model source1 ────────────────────────
    if args.force or not model_source1.is_file():
        print(f"\n{'='*60}", flush=True)
        print("Phase 0/7: Selecting validation IDs + deterministic training sample", flush=True)
        prepare_model_source1(
            train_sources[0], train_truth, valid_truth,
            model_source1, args.train_sample_modulus,
        )

    model_train_sources = [model_source1, train_sources[1], train_sources[2]]

    # ── Phase 1: Training candidates ──────────────────────────
    if not args.skip_train_candidates and (args.force or not train_candidates.is_file()):
        print(f"\n{'='*60}", flush=True)
        print("Phase 1/7: Ultra-high-recall training candidate generation", flush=True)
        build_ultra_recall_candidates(
            *model_train_sources, train_candidates,
            temp / "train_cands",
            threads=args.threads,
            memory_limit=args.memory_limit,
            max_candidates=args.max_candidates,
        )

    # ── Phase 2: Training features ────────────────────────────
    if not args.skip_train_features and (args.force or not train_features.is_file()):
        print(f"\n{'='*60}", flush=True)
        print("Phase 2/7: Building rich 60+ feature training pairs", flush=True)
        build_rich_features(
            train_candidates, *model_train_sources, train_features,
            temp / "train_feats",
            threads=args.threads,
            memory_limit=args.memory_limit,
            train_truth=train_truth,
            valid_truth=valid_truth,
        )

    # ── Phase 3: Train model ───────────────────────────────────
    if not args.skip_train_model:
        print(f"\n{'='*60}", flush=True)
        print("Phase 3/7: Training champion model (LightGBM ensemble or improved logistic)", flush=True)
        model_dict, metrics = train_champion_model(
            train_features, valid_truth, model_path, metrics_path,
            seed=args.seed,
            max_train_rows=args.max_train_rows,
            n_seeds=args.n_seeds,
            n_folds=args.n_folds,
        )
    else:
        print("  Phase 3 skipped — loading existing model", flush=True)
        model_dict = json.loads(model_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}

    if args.validation_only:
        report = {
            "pipeline": "champion_model",
            "mode": "validation_only",
            "elapsed_minutes": round((time.perf_counter() - started) / 60, 2),
            "validation_metrics": metrics,
        }
        (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return

    # ── Phase 4: Test candidates ───────────────────────────────
    if not args.skip_test_candidates and (args.force or not test_candidates.is_file()):
        print(f"\n{'='*60}", flush=True)
        print("Phase 4/7: Generating test candidates (ultra-high-recall blocking)", flush=True)
        build_ultra_recall_candidates(
            *test_sources, test_candidates,
            temp / "test_cands",
            threads=args.threads,
            memory_limit=args.memory_limit,
            max_candidates=args.max_candidates,
        )

    # ── Phase 5: Test features ─────────────────────────────────
    if not args.skip_test_features and (args.force or not test_features.is_file()):
        print(f"\n{'='*60}", flush=True)
        print("Phase 5/7: Building test pair features", flush=True)
        build_rich_features(
            test_candidates, *test_sources, test_features,
            temp / "test_feats",
            threads=args.threads,
            memory_limit=args.memory_limit,
        )

    # ── Phase 6: Score ─────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("Phase 6/7: Scoring test pairs with champion model", flush=True)
    load_and_score(test_features, predictions, model_dict)

    # ── Phase 7: Write submission ──────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("Phase 7/7: Writing final submission", flush=True)
    threshold = float(model_dict.get("threshold", metrics.get("threshold", 0.5)))
    submission_stats = write_submission(
        test_sources[0], predictions, matching, candidate_tsv_out,
        threshold, args.max_matches,
    )

    elapsed = round((time.perf_counter() - started) / 60, 2)
    report = {
        "pipeline": "champion_model",
        "elapsed_minutes": elapsed,
        "validation_metrics": metrics,
        "submission_stats": submission_stats,
        "matching_results": str(matching),
        "candidate_pairs": str(candidate_tsv_out),
        "model": str(model_path),
    }
    (output / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{'='*60}", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print(f"\nDone in {elapsed} minutes.", flush=True)


if __name__ == "__main__":
    main()
