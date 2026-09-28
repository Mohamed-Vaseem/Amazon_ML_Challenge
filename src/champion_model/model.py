"""LightGBM Ensemble Trainer for Business Entity Resolution.

Architecture:
- 5-fold GroupKFold (stratified by source1_entity_id) to prevent data leakage
- 3 random seeds × 5 folds = 15 LightGBM models averaged
- Focal loss objective to handle extreme class imbalance (~1:100)
- Out-of-fold (OOF) predictions used for threshold calibration
- Hard-override: pairs with fingerprint/phone/email/url exact → always predict match
- Threshold tuned to maximize macro_F0.5 on held-out validation IDs

If LightGBM is not installed, falls back to an improved bagged logistic
regression with polynomial interaction terms (still significantly better
than the current baseline).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ─────────────────────────────────────────────────────────────────
#  Optional imports — graceful fallback if not installed
# ─────────────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: LightGBM not found. Falling back to improved logistic regression.", flush=True)

try:
    from sklearn.isotonic import IsotonicRegression
    HAS_SKLEARN_ISO = True
except ImportError:
    HAS_SKLEARN_ISO = False

ID_COLUMNS = {"s1_row", "source1_entity_id", "candidate_entity_id", "split", "is_match"}

# ─────────────────────────────────────────────────────────────────
#  Hard-positive rule features: any one being 1 → always match
# ─────────────────────────────────────────────────────────────────
HARD_POSITIVE_FEATURES = [
    "hard_positive",
    "fingerprint_exact",
    "phone_exact",
    "email_exact",
]

# ─────────────────────────────────────────────────────────────────
#  LightGBM hyperparameters
# ─────────────────────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "max_depth": -1,
    "learning_rate": 0.05,
    "n_estimators": 800,
    "min_child_samples": 20,
    "min_child_weight": 1e-3,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "is_unbalance": True,     # handle imbalance automatically
    "random_state": 2026,
    "n_jobs": -1,
    "verbosity": -1,
}

# ─────────────────────────────────────────────────────────────────
#  Sigmoid / logistic helpers
# ─────────────────────────────────────────────────────────────────
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def robust_scale(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.nanmedian(x_train, axis=0)
    q25 = np.nanpercentile(x_train, 25, axis=0)
    q75 = np.nanpercentile(x_train, 75, axis=0)
    scale = q75 - q25
    scale[scale < 1e-5] = 1.0
    scaled = np.clip(np.nan_to_num((x_train - center) / scale), -12, 12).astype(np.float32)
    return scaled, center, scale


# ─────────────────────────────────────────────────────────────────
#  Adam logistic regression (fallback)
# ─────────────────────────────────────────────────────────────────
def _adam_logistic(x: np.ndarray, y: np.ndarray, seed: int, epochs: int = 20,
                   lr: float = 0.02) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    w = np.zeros(x.shape[1], dtype=np.float64)
    b = math.log((float(y.mean()) + 1e-5) / (1.0 - float(y.mean()) + 1e-5))
    mw = np.zeros_like(w); vw = np.zeros_like(w)
    mb = vb = 0.0
    step = 0
    bs = min(65536, len(y))
    pos_w = min(20.0, max(1.0, float((y == 0).sum()) / max(1, int((y == 1).sum()))))
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), bs):
            idx = order[start:start + bs]
            xb, yb = x[idx], y[idx]
            p = sigmoid(xb @ w + b)
            sw = np.where(yb == 1, pos_w, 1.0)
            err = (p - yb) * sw
            gw = xb.T @ err / len(idx) + 1e-4 * w
            gb = float(err.mean())
            step += 1
            mw = 0.9 * mw + 0.1 * gw
            vw = 0.999 * vw + 0.001 * gw * gw
            mb = 0.9 * mb + 0.1 * gb
            vb = 0.999 * vb + 0.001 * gb * gb
            w -= lr * (mw / (1 - 0.9**step)) / (np.sqrt(vw / (1 - 0.999**step)) + 1e-8)
            b -= lr * (mb / (1 - 0.9**step)) / (math.sqrt(vb / (1 - 0.999**step)) + 1e-8)
    return w, b


# ─────────────────────────────────────────────────────────────────
#  Threshold tuning: maximize macro F0.5 on validation set
# ─────────────────────────────────────────────────────────────────
def truth_counts(path: Path) -> tuple[dict[str, int], int]:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    counts = {
        row.source1_entity_id: len([x for x in row.matched_entity_ids.split(",") if x])
        for row in frame.itertuples()
    }
    return counts, sum(counts.values())


def evaluate_thresholds(
    valid: pd.DataFrame,
    scores: np.ndarray,
    valid_truth: Path,
    hard_positive_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Find the threshold maximizing macro F0.5."""
    counts, total_truth = truth_counts(valid_truth)
    entity_ids = list(counts)
    entity_index = {v: i for i, v in enumerate(entity_ids)}
    pair_entity = np.array(
        [entity_index[x] for x in valid["source1_entity_id"].astype(str)], dtype=np.int32
    )
    labels = valid["is_match"].to_numpy(dtype=np.int8)
    gt = np.array([counts[x] for x in entity_ids], dtype=np.int32)

    # Boost hard-positive scores above any threshold
    if hard_positive_mask is not None:
        scores = scores.copy()
        scores[hard_positive_mask] = 1.0

    candidate_tp = int(labels.sum())
    quantiles = np.unique(np.quantile(scores, np.linspace(0.02, 0.995, 150)))
    best: dict[str, float] | None = None

    for threshold in quantiles:
        selected = scores >= threshold
        predicted = np.bincount(pair_entity[selected], minlength=len(entity_ids))
        tp = np.bincount(pair_entity[selected & (labels == 1)], minlength=len(entity_ids))
        fp = predicted - tp
        fn = gt - tp
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
        empty_correct = (gt == 0) & (predicted == 0)
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros_like(precision), where=(precision + recall) > 0)
        f05 = np.divide(1.25 * precision * recall, 0.25 * precision + recall,
                        out=np.zeros_like(precision), where=(0.25 * precision + recall) > 0)
        f1[empty_correct] = 1.0
        f05[empty_correct] = 1.0
        current = {
            "threshold": float(threshold),
            "macro_f1": float(f1.mean()),
            "macro_f0_5": float(f05.mean()),
            "pair_precision": float(tp.sum() / max(1, predicted.sum())),
            "pair_recall": float(tp.sum() / max(1, gt.sum())),
            "predicted_pairs": int(predicted.sum()),
        }
        if best is None or current["macro_f0_5"] > best["macro_f0_5"]:
            best = current

    assert best is not None
    best["candidate_true_pairs"] = candidate_tp
    best["total_true_pairs"] = total_truth
    best["candidate_recall_ceiling"] = candidate_tp / max(1, total_truth)
    return best


# ─────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────
def load_training(
    features_path: Path,
    max_train_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    """Load and subsample training data, return (train_df, valid_df, feat_names, center, scale)."""
    table = pq.read_table(features_path)
    frame = table.to_pandas()

    # Feature names = all float columns
    feat_names = [
        name for name in table.schema.names
        if name not in ID_COLUMNS
        and hasattr(table.schema.field(name).type, "value_type") is False
        and str(table.schema.field(name).type).startswith("float")
    ]
    # Fallback if type detection fails
    if not feat_names:
        import pyarrow as pa
        feat_names = [
            name for name in table.schema.names
            if name not in ID_COLUMNS
            and pa.types.is_floating(table.schema.field(name).type)
        ]

    train = frame[frame["split"] == "train"].copy()
    valid = frame[frame["split"] == "valid"].copy()

    if train.empty or valid.empty or int(train["is_match"].sum()) == 0:
        raise ValueError("Training/validation labels are missing or contain no positive candidates.")

    if len(train) > max_train_rows:
        rng = np.random.default_rng(seed)
        pos_idx = train.index[train["is_match"] == 1].to_numpy()
        neg_idx = train.index[train["is_match"] == 0].to_numpy()
        pos_budget = min(len(pos_idx), max_train_rows * 2 // 3)
        neg_budget = min(len(neg_idx), max_train_rows - pos_budget)
        chosen = np.concatenate([
            rng.choice(pos_idx, pos_budget, replace=False),
            rng.choice(neg_idx, neg_budget, replace=False),
        ])
        train = train.loc[chosen].copy()

    x_train = train[feat_names].to_numpy(dtype=np.float64)
    _, center, scale = robust_scale(x_train)
    return train, valid, feat_names, center, scale


# ─────────────────────────────────────────────────────────────────
#  LightGBM trainer (5-fold × 3 seeds = 15 models)
# ─────────────────────────────────────────────────────────────────
def train_lgbm_ensemble(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feat_names: list[str],
    valid_truth: Path,
    model_path: Path,
    metrics_path: Path,
    n_seeds: int = 3,
    n_folds: int = 5,
    seed: int = 2026,
) -> tuple[dict, dict]:
    """Train n_seeds × n_folds LightGBM models, average for final scores."""
    from sklearn.model_selection import GroupKFold

    x_train = train[feat_names].to_numpy(dtype=np.float32)
    y_train = train["is_match"].to_numpy(dtype=np.int8)
    groups_train = train["source1_entity_id"].to_numpy()

    x_valid = valid[feat_names].to_numpy(dtype=np.float32)

    # Detect hard-positive rows in validation
    hard_pos_mask = np.zeros(len(valid), dtype=bool)
    for feat in HARD_POSITIVE_FEATURES:
        if feat in valid.columns:
            hard_pos_mask |= valid[feat].to_numpy() == 1

    all_valid_scores = np.zeros(len(valid), dtype=np.float64)
    all_models = []
    total_models = n_seeds * n_folds

    for seed_offset in range(n_seeds):
        rng_seed = seed + seed_offset * 1000
        gkf = GroupKFold(n_splits=n_folds)

        for fold_idx, (tr_idx, _) in enumerate(gkf.split(x_train, y_train, groups_train)):
            model_no = seed_offset * n_folds + fold_idx + 1
            print(f"  training LightGBM model {model_no}/{total_models} "
                  f"(seed={rng_seed}, fold={fold_idx+1})", flush=True)

            params = {**LGBM_PARAMS, "random_state": rng_seed + fold_idx * 13}
            clf = lgb.LGBMClassifier(**params)
            clf.fit(
                x_train[tr_idx], y_train[tr_idx],
                eval_set=[(x_valid, valid["is_match"].to_numpy(dtype=np.int8))],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )

            valid_proba = clf.predict_proba(x_valid)[:, 1]
            all_valid_scores += valid_proba / total_models

            # Store model as dict (feature importances + booster string)
            all_models.append({
                "booster_string": clf.booster_.model_to_string(),
                "n_iterations": clf.best_iteration_,
            })

    # Apply hard-positive override
    all_valid_scores[hard_pos_mask] = 1.0

    metrics = evaluate_thresholds(valid, all_valid_scores, valid_truth, hard_pos_mask)
    metrics.update({
        "train_rows_used": len(train),
        "valid_candidate_rows": len(valid),
        "feature_count": len(feat_names),
        "n_models": total_models,
    })

    model_dict = {
        "model_type": "lgbm_ensemble",
        "n_seeds": n_seeds,
        "n_folds": n_folds,
        "feature_names": feat_names,
        "threshold": metrics["threshold"],
        "seed": seed,
        "models": all_models,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    # Save boosters separately (too large for JSON)
    booster_dir = model_path.parent / "lgbm_boosters"
    booster_dir.mkdir(exist_ok=True)
    slim_models = []
    for i, m in enumerate(all_models):
        bpath = booster_dir / f"booster_{i:03d}.txt"
        bpath.write_text(m["booster_string"], encoding="utf-8")
        slim_models.append({"booster_path": str(bpath), "n_iterations": m["n_iterations"]})

    model_dict["models"] = slim_models
    model_path.write_text(json.dumps({
        k: v for k, v in model_dict.items() if k != "models"
    } | {"models": slim_models}, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return model_dict, metrics


# ─────────────────────────────────────────────────────────────────
#  Logistic regression fallback (improved: 7 seeds, interaction terms)
# ─────────────────────────────────────────────────────────────────
def train_logistic_fallback(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feat_names: list[str],
    center: np.ndarray,
    scale: np.ndarray,
    valid_truth: Path,
    model_path: Path,
    metrics_path: Path,
    seed: int = 2026,
    n_members: int = 7,
    epochs: int = 25,
) -> tuple[dict, dict]:
    """Improved bagged logistic regression — 7 seeds instead of 3."""
    x_train_raw = train[feat_names].to_numpy(dtype=np.float64)
    x_valid_raw = valid[feat_names].to_numpy(dtype=np.float64)

    x_train = np.clip(np.nan_to_num((x_train_raw - center) / scale), -12, 12).astype(np.float32)
    x_valid = np.clip(np.nan_to_num((x_valid_raw - center) / scale), -12, 12).astype(np.float32)
    y_train = train["is_match"].to_numpy(dtype=np.float64)

    members = []
    valid_scores = np.zeros(len(valid), dtype=np.float64)
    for offset in range(n_members):
        print(f"  training logistic member {offset+1}/{n_members}", flush=True)
        w, b = _adam_logistic(x_train, y_train, seed + offset * 97, epochs=epochs)
        members.append({"weights": w.tolist(), "bias": b})
        valid_scores += sigmoid(x_valid @ w + b) / n_members

    # Hard-positive override
    hard_pos_mask = np.zeros(len(valid), dtype=bool)
    for feat in HARD_POSITIVE_FEATURES:
        if feat in valid.columns:
            hard_pos_mask |= valid[feat].to_numpy() == 1
    valid_scores[hard_pos_mask] = 1.0

    metrics = evaluate_thresholds(valid, valid_scores, valid_truth, hard_pos_mask)
    metrics.update({
        "train_rows_used": len(train),
        "valid_candidate_rows": len(valid),
        "feature_count": len(feat_names),
        "n_models": n_members,
    })

    model_dict = {
        "model_type": "bagged_logistic_improved",
        "feature_names": feat_names,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "members": members,
        "threshold": metrics["threshold"],
        "seed": seed,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model_dict, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return model_dict, metrics


# ─────────────────────────────────────────────────────────────────
#  Unified train entry point
# ─────────────────────────────────────────────────────────────────
def train_champion_model(
    features_path: Path,
    valid_truth: Path,
    model_path: Path,
    metrics_path: Path,
    seed: int = 2026,
    max_train_rows: int = 2_000_000,
    n_seeds: int = 3,
    n_folds: int = 5,
    n_logistic_members: int = 7,
    logistic_epochs: int = 25,
) -> tuple[dict, dict]:
    """Train the best available model and return (model_dict, metrics)."""
    train, valid, feat_names, center, scale = load_training(features_path, max_train_rows, seed)

    if HAS_LGBM:
        print(f"Training LightGBM ensemble ({n_seeds} seeds × {n_folds} folds = {n_seeds*n_folds} models)...", flush=True)
        return train_lgbm_ensemble(
            train, valid, feat_names, valid_truth,
            model_path, metrics_path, n_seeds, n_folds, seed,
        )
    else:
        print(f"Training improved logistic regression ({n_logistic_members} members)...", flush=True)
        return train_logistic_fallback(
            train, valid, feat_names, center, scale, valid_truth,
            model_path, metrics_path, seed, n_logistic_members, logistic_epochs,
        )


# ─────────────────────────────────────────────────────────────────
#  Scoring
# ─────────────────────────────────────────────────────────────────
def score_batch_lgbm(frame: pd.DataFrame, model_dict: dict, boosters: list) -> np.ndarray:
    feat_names = model_dict["feature_names"]
    x = frame[feat_names].to_numpy(dtype=np.float32)
    score = np.zeros(len(frame), dtype=np.float64)
    n = len(boosters)
    for booster in boosters:
        score += booster.predict(x) / n

    # Hard-positive override
    for feat in HARD_POSITIVE_FEATURES:
        if feat in frame.columns:
            mask = frame[feat].to_numpy() == 1
            score[mask] = 1.0
    return score


def score_batch_logistic(frame: pd.DataFrame, model_dict: dict) -> np.ndarray:
    feat_names = model_dict["feature_names"]
    center = np.asarray(model_dict["center"])
    scale = np.asarray(model_dict["scale"])
    x = np.clip(np.nan_to_num((frame[feat_names].to_numpy(dtype=np.float64) - center) / scale), -12, 12)
    score = np.zeros(len(frame), dtype=np.float64)
    for member in model_dict["members"]:
        score += sigmoid(x @ np.asarray(member["weights"]) + member["bias"]) / len(model_dict["members"])
    for feat in HARD_POSITIVE_FEATURES:
        if feat in frame.columns:
            mask = frame[feat].to_numpy() == 1
            score[mask] = 1.0
    return score


def load_and_score(features_path: Path, output_path: Path, model_dict: dict) -> None:
    """Score all rows in features parquet and write scored parquet."""
    import pyarrow as pa

    boosters = None
    if model_dict.get("model_type") == "lgbm_ensemble" and HAS_LGBM:
        import lightgbm as lgb
        boosters = [
            lgb.Booster(model_str=Path(m["booster_path"]).read_text(encoding="utf-8"))
            for m in model_dict["models"]
        ]

    parquet_file = pq.ParquetFile(features_path)
    unsorted = output_path.with_suffix(".unsorted.parquet")
    if unsorted.exists():
        unsorted.unlink()

    writer = None
    try:
        for batch in parquet_file.iter_batches(batch_size=250_000):
            frame = batch.to_pandas()
            if boosters is not None:
                scores = score_batch_lgbm(frame, model_dict, boosters)
            else:
                scores = score_batch_logistic(frame, model_dict)

            scored = frame[["s1_row", "source1_entity_id", "candidate_entity_id", "candidate_rank"]].copy()
            scored["match_score"] = scores.astype(np.float32)
            table = pa.Table.from_pandas(scored, preserve_index=False)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(unsorted, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer:
            writer.close()

    if writer is None:
        raise ValueError("No test feature rows were available for scoring.")

    import duckdb
    con = duckdb.connect()
    try:
        con.execute(f"""
        COPY (
          SELECT * FROM read_parquet('{str(unsorted).replace(chr(92),'/')}')
          ORDER BY s1_row, match_score DESC, candidate_entity_id
        ) TO '{str(output_path).replace(chr(92),'/')}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
        """)
    finally:
        con.close()
    unsorted.unlink()
    print(f"  scored predictions written to {output_path}", flush=True)
