"""
FAST Pipeline - Completes in ~90 minutes instead of 4+ hours.
Key speedups:
  - 3 models (3-fold x 1 seed) instead of 10
  - NO bootstrap CI (saves 20-30 min)
  - Phase 5 in micro-chunks with 2 threads (avoids OOM)
  - Phase 6 scoring in streaming batches
"""
from __future__ import annotations
import json, os, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
import lightgbm as lgb

# Paths
DATA     = Path("data/processed")
SPLIT    = Path("data/splits")
WORK     = Path("data/advanced_hybrid")
OUTPUT   = Path("output/advanced_hybrid")
MODEL    = Path("models/advanced_hybrid")

WORK.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
MODEL.mkdir(parents=True, exist_ok=True)

train_feats  = WORK / "train_features.parquet"
test_cands   = WORK / "test_candidates.parquet"
test_feats   = WORK / "test_features.parquet"
test_preds   = WORK / "test_predictions.parquet"
test_s1      = DATA / "test_source1_clean.parquet"
test_s2      = DATA / "test_source2_clean.parquet"
test_s3      = DATA / "test_source3_clean.parquet"
valid_truth  = SPLIT / "valid_ground_truth.tsv"

HARD_POSITIVE_FEATURES = ["hard_positive", "fingerprint_exact", "phone_exact", "email_exact", "url_exact"]

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "max_depth": -1,
    "learning_rate": 0.05,
    "n_estimators": 600,
    "min_child_samples": 20,
    "min_child_weight": 1e-3,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "is_unbalance": True,
    "random_state": 2026,
    "n_jobs": -1,
    "verbosity": -1,
}


def sql_path(p):
    return str(p).replace("\\", "/")


def numeric_feature_names(schema):
    from src.advanced_hybrid.features import numeric_feature_names as _nfn
    return _nfn(schema)


def fast_train():
    from sklearn.model_selection import GroupKFold
    from src.advanced_hybrid.model import truth_counts, evaluate_threshold_grid

    t0 = time.perf_counter()
    print("=" * 70, flush=True)
    print(" FAST PIPELINE - 3 models, no bootstrap", flush=True)
    print("=" * 70, flush=True)

    meta_path = MODEL / "model_metadata.json"
    booster_dir = MODEL / "lgbm_boosters"
    if meta_path.exists():
        print("\n[CACHED] Loading saved model metadata...", flush=True)
        model_metadata = json.loads(meta_path.read_text())
        all_exist = all(Path(e["booster_path"]).exists() for e in model_metadata["boosters"])
        if all_exist:
            print(f"  {len(model_metadata['boosters'])} boosters found, threshold={model_metadata['threshold']:.5f}", flush=True)
            return model_metadata
        print("  Some boosters missing, retraining...", flush=True)

    print("\nPhase 3-FAST: Training 3 LightGBM models...", flush=True)

    pf = pq.ParquetFile(train_feats)
    feat_cols = numeric_feature_names(pf.schema_arrow)
    print(f"  feature count: {len(feat_cols)}", flush=True)

    needed_cols = feat_cols + ["split", "is_match", "source1_entity_id", "candidate_entity_id"]
    cols_sql = ", ".join(needed_cols)

    con = duckdb.connect()
    try:
        con.execute("SET threads = 4")
        con.execute("SET memory_limit = '7GB'")
        con.execute(f"SET temp_directory = '{sql_path(WORK / 'tmp_train_feats')}'")

        print("  loading validation features...", flush=True)
        valid_df = con.execute(f"""
            SELECT {cols_sql}
            FROM read_parquet('{sql_path(train_feats)}')
            WHERE split = 'valid'
        """).df()

        print("  loading and subsampling train features...", flush=True)
        pos_count = con.execute(f"""
            SELECT count(*)
            FROM read_parquet('{sql_path(train_feats)}')
            WHERE split = 'train' AND is_match = 1
        """).fetchone()[0]

        max_train = 1_200_000
        sample_n = max(10, max_train - pos_count)

        train_df = con.execute(f"""
            WITH train_all AS (
                SELECT {cols_sql}
                FROM read_parquet('{sql_path(train_feats)}')
                WHERE split = 'train'
            ),
            positives AS (SELECT * FROM train_all WHERE is_match = 1),
            negatives AS (SELECT * FROM train_all WHERE is_match = 0),
            sampled_negatives AS (
                SELECT * FROM negatives
                USING SAMPLE reservoir({sample_n} ROWS)
                REPEATABLE (2026)
            )
            SELECT * FROM positives
            UNION ALL
            SELECT * FROM sampled_negatives
        """).df()
    finally:
        con.close()

    train_df = train_df.sample(frac=1.0, random_state=2026).reset_index(drop=True)
    for col in feat_cols:
        train_df[col] = train_df[col].astype(np.float32)
        valid_df[col] = valid_df[col].astype(np.float32)

    pos_c = int(train_df["is_match"].sum())
    print(f"  train: {len(train_df):,} ({pos_c:,} pos) | valid: {len(valid_df):,}", flush=True)

    n_folds = 3
    x_train = train_df[feat_cols].to_numpy(dtype=np.float32)
    y_train = train_df["is_match"].to_numpy(dtype=np.int8)
    groups = train_df["source1_entity_id"].to_numpy()
    x_valid = valid_df[feat_cols].to_numpy(dtype=np.float32)

    all_valid_scores = np.zeros(len(valid_df), dtype=np.float64)
    model_entries = []
    booster_dir.mkdir(parents=True, exist_ok=True)

    gkf = GroupKFold(n_splits=n_folds)
    for fold, (trn_idx, _) in enumerate(gkf.split(x_train, y_train, groups)):
        print(f"  training model {fold+1}/{n_folds}...", flush=True)
        params = {**LGBM_PARAMS, "random_state": 2026 + fold * 17}
        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            x_train[trn_idx], y_train[trn_idx],
            eval_set=[(x_valid, valid_df["is_match"].to_numpy(dtype=np.int8))],
            callbacks=[
                lgb.early_stopping(stopping_rounds=40, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        val_probs = clf.predict_proba(x_valid)[:, 1]
        all_valid_scores += val_probs / n_folds

        bpath = booster_dir / f"booster_{fold+1:03d}.txt"
        bpath.write_text(clf.booster_.model_to_string(), encoding="utf-8")
        model_entries.append({
            "booster_path": str(bpath),
            "n_iterations": clf.best_iteration_,
        })

    for hp_col in HARD_POSITIVE_FEATURES:
        if hp_col in valid_df.columns:
            all_valid_scores[valid_df[hp_col].to_numpy() == 1] = 1.0

    print("  calibrating threshold...", flush=True)
    truth_map, _ = truth_counts(valid_truth)
    eval_metrics = evaluate_threshold_grid(valid_df, all_valid_scores, truth_map, max_matches=11)
    best_thresh = eval_metrics["best_threshold"]

    print(f"  Macro F0.5: {eval_metrics['macro_f0_5']:.5f} @ threshold={best_thresh:.4f}", flush=True)
    print(f"  Phase 3 done in {(time.perf_counter()-t0)/60:.1f} min", flush=True)

    model_metadata = {
        "model_type": "lgbm_ensemble",
        "n_seeds": 1,
        "n_folds": n_folds,
        "feature_names": feat_cols,
        "threshold": best_thresh,
        "max_matches": 11,
        "boosters": model_entries,
        "validation_metrics": eval_metrics,
    }
    meta_path.write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")
    return model_metadata


def fast_test_features(model_metadata):
    from src.advanced_hybrid.features import build_advanced_features

    if test_feats.exists():
        print(f"\n[CACHED] test features -> {test_feats}", flush=True)
        return

    t0 = time.perf_counter()
    print("\nPhase 5-FAST: Building test features...", flush=True)

    temp_dir = WORK / "tmp_test_feats"
    temp_dir.mkdir(parents=True, exist_ok=True)

    build_advanced_features(
        candidates=test_cands,
        source1=test_s1,
        source2=test_s2,
        source3=test_s3,
        output=test_feats,
        temp_dir=temp_dir,
        threads=2,
        memory_limit="7GB",
    )
    print(f"  Phase 5 done in {(time.perf_counter()-t0)/60:.1f} min", flush=True)


def fast_score(model_metadata):
    if test_preds.exists():
        print(f"\n[CACHED] test predictions -> {test_preds}", flush=True)
        return

    t0 = time.perf_counter()
    print("\nPhase 6-FAST: Scoring test pairs...", flush=True)

    from src.advanced_hybrid.model import score_features_file
    score_features_file(
        features_parquet=test_feats,
        output_predictions_parquet=test_preds,
        model_metadata=model_metadata,
    )
    print(f"  Phase 6 done in {(time.perf_counter()-t0)/60:.1f} min", flush=True)


def fast_submission(model_metadata):
    t0 = time.perf_counter()
    print("\nPhase 7-FAST: Writing submission files...", flush=True)

    from src.advanced_hybrid.model import write_submission_files

    threshold = model_metadata["threshold"]
    final_matching = OUTPUT / "matching_results.tsv"
    final_candidates = OUTPUT / "candidate_pairs.tsv"

    sub_stats = write_submission_files(
        source1_clean_parquet=test_s1,
        predictions_parquet=test_preds,
        matching_tsv_path=final_matching,
        candidate_tsv_path=final_candidates,
        threshold=threshold,
        max_matches=11,
    )

    result_tsv = Path("result.tsv")
    shutil.copy2(final_matching, result_tsv)

    print(f"\n  Submission stats: {json.dumps(sub_stats, indent=2)}", flush=True)
    print(f"  matching_results.tsv -> {final_matching}", flush=True)
    print(f"  candidate_pairs.tsv  -> {final_candidates}", flush=True)
    print(f"  result.tsv           -> {result_tsv.absolute()}", flush=True)
    print(f"  Phase 7 done in {(time.perf_counter()-t0)/60:.1f} min", flush=True)
    return sub_stats


if __name__ == "__main__":
    start = time.perf_counter()

    assert train_feats.exists(), f"Missing: {train_feats}"
    assert test_cands.exists(),  f"Missing: {test_cands}"

    model_metadata = fast_train()
    fast_test_features(model_metadata)
    fast_score(model_metadata)
    sub_stats = fast_submission(model_metadata)

    elapsed = (time.perf_counter() - start) / 60
    print("\n" + "=" * 70, flush=True)
    print(f" FAST PIPELINE COMPLETE in {elapsed:.1f} minutes", flush=True)
    print("=" * 70, flush=True)
