"""Score candidate pairs and write final challenge-style candidate pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ID_COLUMNS = ["source1_entity_id", "candidate_entity_id"]


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def load_model(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def score_frame(frame: pd.DataFrame, model: dict[str, object]) -> pd.DataFrame:
    feature_names = list(model["feature_names"])
    center = np.array(model["center"], dtype="float64")
    scale = np.array(model["scale"], dtype="float64")
    weights = np.array(model["weights"], dtype="float64")
    for name in feature_names:
        if name not in frame.columns:
            frame[name] = 0.0
    raw = frame[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    x = np.nan_to_num((raw - center) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    out = frame.copy()
    out["match_score"] = sigmoid(x @ weights).astype("float32")
    out["predicted_match"] = out["match_score"].ge(float(model["threshold"])).astype("int8")
    return out


def write_submission(prediction_path: Path, submission_path: Path, threshold: float) -> None:
    frame = pq.read_table(prediction_path, columns=["source1_entity_id", "candidate_entity_id", "match_score"]).to_pandas()
    keep = frame[frame["match_score"] >= threshold].copy()
    keep = keep.sort_values(["source1_entity_id", "match_score"], ascending=[True, False])
    grouped = (
        keep.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda values: ",".join(dict.fromkeys(values.astype(str))))
        .to_dict()
    )
    all_ids = frame["source1_entity_id"].astype(str).drop_duplicates().tolist()
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    with submission_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("source1_entity_id\tcandidate_entity_ids\n")
        for source1_id in all_ids:
            handle.write(f"{source1_id}\t{grouped.get(source1_id, '')}\n")


def predict(features_path: Path, model_path: Path, predictions_path: Path, submission_path: Path | None) -> dict[str, object]:
    model = load_model(model_path)
    parquet = pq.ParquetFile(features_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    predicted_positive = 0
    try:
        for batch in parquet.iter_batches(batch_size=500_000):
            frame = batch.to_pandas()
            scored = score_frame(frame, model)
            rows += len(scored)
            predicted_positive += int(scored["predicted_match"].sum())
            table = pa.Table.from_pandas(scored, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(predictions_path, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        pd.DataFrame(columns=ID_COLUMNS + ["match_score", "predicted_match"]).to_parquet(predictions_path, index=False)

    if submission_path is not None:
        write_submission(predictions_path, submission_path, float(model["threshold"]))

    summary = {
        "features": str(features_path),
        "model": str(model_path),
        "predictions": str(predictions_path),
        "submission": str(submission_path) if submission_path else None,
        "rows": rows,
        "predicted_positive": predicted_positive,
        "threshold": float(model["threshold"]),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Team B pair features.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--submission")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predict(
        Path(args.features),
        Path(args.model),
        Path(args.predictions),
        Path(args.submission) if args.submission else None,
    )


if __name__ == "__main__":
    main()

