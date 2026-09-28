"""Train a compact pair scorer from validation pair features.

This module intentionally depends only on numpy/pandas/pyarrow so it can run in
the current challenge environment. It learns a calibrated linear scorer by
searching weighted feature combinations and picking the threshold that maximizes
F1 on the validation split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EXCLUDE_COLUMNS = {
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "country",
    "blocking_methods",
    "is_match",
}


def load_features(path: Path) -> pd.DataFrame:
    frame = pq.read_table(path).to_pandas()
    if "is_match" not in frame.columns:
        raise ValueError("Training features must contain is_match labels.")
    if frame["is_match"].sum() == 0:
        raise ValueError("No positive labels found. Check validation candidates and ground truth.")
    return frame


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if column in EXCLUDE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    if not columns:
        raise ValueError("No numeric feature columns found.")
    return columns


def robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = values.astype("float64", copy=False)
    center = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    scale = q75 - q25
    scale[scale < 1e-6] = 1.0
    scaled = np.nan_to_num((values - center) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    return scaled, center, scale


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= threshold
    positives = y_true == 1
    tp = int(np.sum(pred & positives))
    fp = int(np.sum(pred & ~positives))
    fn = int(np.sum(~pred & positives))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive": int(np.sum(pred)),
    }


def tune_threshold(y_true: np.ndarray, scores: np.ndarray, recall_floor: float = 0.90) -> dict[str, float]:
    unique_scores = np.unique(np.quantile(scores, np.linspace(0.001, 0.999, 500)))
    best = {"f1": -1.0, "recall": 0.0, "threshold": 0.5}
    best_floor = None
    for threshold in unique_scores:
        current = metrics_at_threshold(y_true, scores, float(threshold))
        if current["f1"] > best["f1"]:
            best = current
        if current["recall"] >= recall_floor:
            if best_floor is None or current["precision"] > best_floor["precision"]:
                best_floor = current
    return best_floor or best


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    positives = int(y_sorted.sum())
    if positives == 0:
        return 0.0
    cumulative_tp = np.cumsum(y_sorted)
    precision = cumulative_tp / (np.arange(len(y_sorted)) + 1)
    return float(np.sum(precision * y_sorted) / positives)


def random_weight_search(x: np.ndarray, y: np.ndarray, feature_names: list[str], seed: int, rounds: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n_features = x.shape[1]

    priors = {
        "blocking_score": 2.0,
        "name_similarity": 2.6,
        "address_similarity": 1.4,
        "token_jaccard": 1.8,
        "number_jaccard": 1.1,
        "postal_match": 1.5,
        "identity_consensus": 3.0,
        "name_address_product": 1.4,
        "reciprocal_rank": 1.1,
        "multi_evidence": 1.2,
    }
    base = np.array([priors.get(name, 0.25) for name in feature_names], dtype="float64")
    base = base / max(base.sum(), 1e-9)

    candidates: list[np.ndarray] = [base]
    for _ in range(rounds):
        noise = rng.gamma(shape=1.2, scale=1.0, size=n_features)
        weights = 0.65 * base + 0.35 * noise / noise.sum()
        signed = weights.copy()
        for name in ("rank_overall", "rank_within_source", "candidate_count", "score_gap_from_best", "source_score_gap"):
            if name in feature_names:
                signed[feature_names.index(name)] *= -1.0
        candidates.append(signed)

    best: dict[str, object] | None = None
    for weights in candidates:
        raw = x @ weights
        scores = sigmoid(raw)
        threshold_metrics = tune_threshold(y, scores)
        ap = average_precision(y, scores)
        objective = threshold_metrics["f1"] + 0.05 * ap
        if best is None or objective > best["objective"]:
            best = {
                "objective": float(objective),
                "weights": weights.tolist(),
                "threshold_metrics": threshold_metrics,
                "average_precision": ap,
            }
    assert best is not None
    return best


def train(features_path: Path, model_output: Path, metrics_output: Path, seed: int, rounds: int) -> dict[str, object]:
    frame = load_features(features_path)
    feature_names = numeric_feature_columns(frame)
    y = frame["is_match"].astype("int8").to_numpy()
    x_raw = frame[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    x, center, scale = robust_scale(x_raw)

    best = random_weight_search(x, y, feature_names, seed, rounds)
    model = {
        "model_type": "robust_linear_pair_scorer",
        "feature_names": feature_names,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "weights": best["weights"],
        "threshold": best["threshold_metrics"]["threshold"],
        "training_rows": int(len(frame)),
        "positive_rows": int(y.sum()),
        "seed": seed,
    }
    metrics = {
        "features": str(features_path),
        "model_output": str(model_output),
        "training_rows": int(len(frame)),
        "positive_rows": int(y.sum()),
        "positive_rate": float(y.mean()),
        "average_precision": best["average_precision"],
        **best["threshold_metrics"],
    }

    model_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.write_text(json.dumps(model, indent=2), encoding="utf-8")
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Team B pair scorer from validation features.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--model-output", default="models/team_b_pair_model.json")
    parser.add_argument("--metrics-output", default="output/team_b_model_metrics.json")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rounds", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(Path(args.features), Path(args.model_output), Path(args.metrics_output), args.seed, args.rounds)


if __name__ == "__main__":
    main()

