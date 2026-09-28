"""Create a deterministic validation split for entity-resolution evaluation.

The split is performed at Source 1 entity level.  Matching cardinality and
country are included in the stable hash so that reruns are reproducible and
each important slice receives an independent holdout sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def match_bucket(value: object) -> str:
    text = str(value or "")
    if not text:
        return "0"
    count = text.count(",") + 1
    if count == 1:
        return "1"
    if count <= 3:
        return "2-3"
    return "4+"


def stable_fraction(key: str, seed: int) -> float:
    digest = hashlib.blake2b(f"{seed}:{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64)


def split_ground_truth(
    ground_truth: pd.DataFrame,
    source1_country: pd.Series,
    valid_fraction: float,
    seed: int,
) -> pd.DataFrame:
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be between 0 and 1")

    missing = set(GROUND_TRUTH_COLUMNS) - set(ground_truth.columns)
    if missing:
        raise ValueError(f"Ground truth is missing columns: {sorted(missing)}")

    frame = ground_truth[GROUND_TRUTH_COLUMNS].fillna("").astype(str).copy()
    if frame["source1_entity_id"].duplicated().any():
        examples = frame.loc[frame["source1_entity_id"].duplicated(), "source1_entity_id"].head(5).tolist()
        raise ValueError(f"Duplicate Source 1 IDs in ground truth: {examples}")

    frame["country"] = frame["source1_entity_id"].map(source1_country)
    missing_country = frame["country"].isna()
    if missing_country.any():
        examples = frame.loc[missing_country, "source1_entity_id"].head(5).tolist()
        raise ValueError(f"Ground-truth IDs missing from cleaned Source 1: {examples}")

    frame["match_bucket"] = frame["matched_entity_ids"].map(match_bucket)
    split_keys = (
        frame["country"].astype(str)
        + "\x1f"
        + frame["match_bucket"]
        + "\x1f"
        + frame["source1_entity_id"]
    )
    frame["split_value"] = split_keys.map(lambda key: stable_fraction(key, seed))
    frame["split"] = "train"
    frame.loc[frame["split_value"] < valid_fraction, "split"] = "valid"

    if len(frame) >= 2 and not frame["split"].eq("valid").any():
        frame.loc[frame["split_value"].idxmin(), "split"] = "valid"
    if len(frame) >= 2 and not frame["split"].eq("train").any():
        frame.loc[frame["split_value"].idxmax(), "split"] = "train"
    return frame


def create_split(
    ground_truth_path: Path,
    source1_path: Path,
    output_dir: Path,
    valid_fraction: float,
    seed: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = pd.read_csv(
        ground_truth_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=GROUND_TRUTH_COLUMNS,
    )
    source1 = pd.read_parquet(source1_path, columns=["entity_id", "country"]).fillna("")
    source1["entity_id"] = source1["entity_id"].astype(str)
    if source1["entity_id"].duplicated().any():
        raise ValueError("Cleaned Source 1 contains duplicate entity_id values")

    source1_country = source1.set_index("entity_id")["country"]
    split = split_ground_truth(ground_truth, source1_country, valid_fraction, seed)
    train = split.loc[split["split"].eq("train"), GROUND_TRUTH_COLUMNS]
    valid = split.loc[split["split"].eq("valid"), GROUND_TRUTH_COLUMNS]

    train_path = output_dir / "train_ground_truth.tsv"
    valid_path = output_dir / "valid_ground_truth.tsv"
    train_ids_path = output_dir / "train_source1_ids.txt"
    valid_ids_path = output_dir / "valid_source1_ids.txt"
    train.to_csv(train_path, sep="\t", index=False)
    valid.to_csv(valid_path, sep="\t", index=False)
    train["source1_entity_id"].to_csv(train_ids_path, index=False, header=False)
    valid["source1_entity_id"].to_csv(valid_ids_path, index=False, header=False)

    slice_counts = (
        split.groupby(["split", "country", "match_bucket"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["split", "rows"], ascending=[True, False])
    )
    slice_counts.to_csv(output_dir / "split_slice_counts.tsv", sep="\t", index=False)

    manifest: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "requested_valid_fraction": valid_fraction,
        "actual_valid_fraction": len(valid) / len(split) if len(split) else 0.0,
        "total_rows": len(split),
        "train_rows": len(train),
        "valid_rows": len(valid),
        "train_true_matches": int(train["matched_entity_ids"].map(lambda x: 0 if not x else x.count(",") + 1).sum()),
        "valid_true_matches": int(valid["matched_entity_ids"].map(lambda x: 0 if not x else x.count(",") + 1).sum()),
        "country_count": int(split["country"].nunique()),
        "paths": {
            "train_ground_truth": str(train_path),
            "valid_ground_truth": str(valid_path),
            "train_source1_ids": str(train_ids_path),
            "valid_source1_ids": str(valid_ids_path),
        },
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic train/validation ground-truth splits.")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--source1", required=True, help="Cleaned train_source1 parquet.")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--valid-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = create_split(
        Path(args.ground_truth),
        Path(args.source1),
        Path(args.output_dir),
        args.valid_fraction,
        args.seed,
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
