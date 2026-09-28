"""Build model-ready pair features from blocking candidates in bounded memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PAIR_SEPARATOR = "\x1f"
METHODS = (
    "postal_exact",
    "name_prefix",
    "rare_name_token",
    "address_number",
    "name_minhash_band",
)
INPUT_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "country",
    "blocking_score",
    "blocking_methods",
    "name_similarity",
    "address_similarity",
    "token_jaccard",
    "number_jaccard",
    "postal_match",
]


def read_truth_pairs(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    truth = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    pairs: set[str] = set()
    for row in truth.itertuples(index=False):
        source1_id = str(row.source1_entity_id)
        for candidate_id in str(row.matched_entity_ids).split(","):
            if candidate_id:
                pairs.add(f"{source1_id}{PAIR_SEPARATOR}{candidate_id}")
    return pairs


def engineer_features(frame: pd.DataFrame, truth_pairs: set[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame[INPUT_COLUMNS].fillna("").copy()
    numeric_columns = [
        "blocking_score",
        "name_similarity",
        "address_similarity",
        "token_jaccard",
        "number_jaccard",
        "postal_match",
    ]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).astype("float32")

    out = out.sort_values(
        ["source1_entity_id", "blocking_score", "candidate_entity_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    entity_group = out.groupby("source1_entity_id", sort=False)
    source_group = out.groupby(["source1_entity_id", "candidate_source"], sort=False)
    out["rank_overall"] = entity_group.cumcount().add(1).astype("int16")
    out["rank_within_source"] = source_group.cumcount().add(1).astype("int16")
    out["candidate_count"] = entity_group["candidate_entity_id"].transform("size").astype("int16")
    out["source_candidate_count"] = source_group["candidate_entity_id"].transform("size").astype("int16")

    best_score = entity_group["blocking_score"].transform("max").astype("float32")
    source_best_score = source_group["blocking_score"].transform("max").astype("float32")
    out["score_gap_from_best"] = (best_score - out["blocking_score"]).astype("float32")
    out["source_score_gap"] = (source_best_score - out["blocking_score"]).astype("float32")
    out["score_ratio_to_best"] = np.divide(
        out["blocking_score"],
        best_score,
        out=np.zeros(len(out), dtype=np.float32),
        where=best_score.to_numpy() > 0,
    )
    out["reciprocal_rank"] = (1.0 / out["rank_overall"].astype("float32")).astype("float32")

    method_text = out["blocking_methods"].astype(str)
    for method in METHODS:
        out[f"method_{method}"] = method_text.str.contains(method, regex=False).astype("int8")
    method_columns = [f"method_{method}" for method in METHODS]
    out["blocking_method_count"] = out[method_columns].sum(axis=1).astype("int8")
    out["multi_evidence"] = out["blocking_method_count"].ge(2).astype("int8")
    out["candidate_source_s2"] = out["candidate_source"].eq("S2").astype("int8")

    out["name_address_product"] = (out["name_similarity"] * out["address_similarity"]).astype("float32")
    out["name_token_mean"] = ((out["name_similarity"] + out["token_jaccard"]) / 2.0).astype("float32")
    out["address_number_mean"] = ((out["address_similarity"] + out["number_jaccard"]) / 2.0).astype("float32")
    out["identity_consensus"] = (
        0.40 * out["name_similarity"]
        + 0.20 * out["address_similarity"]
        + 0.15 * out["token_jaccard"]
        + 0.10 * out["number_jaccard"]
        + 0.15 * out["postal_match"]
    ).astype("float32")
    out["strong_name_and_location"] = (
        out["name_similarity"].ge(0.80)
        & (out["address_similarity"].ge(0.65) | out["postal_match"].eq(1.0))
    ).astype("int8")

    if truth_pairs is not None:
        keys = out["source1_entity_id"].astype(str) + PAIR_SEPARATOR + out["candidate_entity_id"].astype(str)
        out["is_match"] = keys.isin(truth_pairs).astype("int8")
    return out


def build_features(
    candidates_path: Path,
    output_path: Path,
    ground_truth_path: Path | None = None,
    batch_size: int = 500_000,
) -> dict[str, object]:
    truth_pairs = read_truth_pairs(ground_truth_path)
    parquet = pq.ParquetFile(candidates_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    carry = pd.DataFrame(columns=INPUT_COLUMNS)
    total_rows = 0
    positive_rows = 0
    feature_columns: list[str] = []

    def write_complete(frame: pd.DataFrame) -> None:
        nonlocal writer, total_rows, positive_rows, feature_columns
        if frame.empty:
            return
        featured = engineer_features(frame, truth_pairs)
        feature_columns = featured.columns.tolist()
        total_rows += len(featured)
        if "is_match" in featured:
            positive_rows += int(featured["is_match"].sum())
        table = pa.Table.from_pandas(featured, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
        writer.write_table(table)
        if total_rows % 5_000_000 < len(featured):
            print(f"built {total_rows:,} feature rows", flush=True)

    try:
        for batch in parquet.iter_batches(batch_size=batch_size, columns=INPUT_COLUMNS):
            frame = batch.to_pandas()
            if not carry.empty:
                frame = pd.concat([carry, frame], ignore_index=True)
            last_id = str(frame.iloc[-1]["source1_entity_id"])
            last_mask = frame["source1_entity_id"].astype(str).eq(last_id)
            write_complete(frame.loc[~last_mask])
            carry = frame.loc[last_mask].copy()
        write_complete(carry)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        empty = engineer_features(pd.DataFrame(columns=INPUT_COLUMNS), truth_pairs)
        empty.to_parquet(output_path, index=False)

    manifest = {
        "candidates": str(candidates_path),
        "output": str(output_path),
        "ground_truth": str(ground_truth_path) if ground_truth_path else None,
        "rows": total_rows,
        "positive_rows": positive_rows if truth_pairs is not None else None,
        "feature_columns": feature_columns,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model-ready pair features in bounded memory.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ground-truth", help="Optional TSV used to add is_match labels.")
    parser.add_argument("--batch-size", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_features(
        Path(args.candidates),
        Path(args.output),
        Path(args.ground_truth) if args.ground_truth else None,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
