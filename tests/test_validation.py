from pathlib import Path

import pandas as pd

from src.validation.create_validation_split import create_split, split_ground_truth
from src.validation.evaluate_validation import evaluate


def test_split_is_deterministic_and_disjoint(tmp_path: Path):
    truth = pd.DataFrame(
        {
            "source1_entity_id": [f"S1-{index}" for index in range(200)],
            "matched_entity_ids": [f"S2-{index}" if index % 4 else "" for index in range(200)],
        }
    )
    country = pd.Series(
        {f"S1-{index}": "IN" if index % 2 else "US" for index in range(200)},
        name="country",
    )

    first = split_ground_truth(truth, country, valid_fraction=0.2, seed=17)
    second = split_ground_truth(truth, country, valid_fraction=0.2, seed=17)

    assert first["split"].tolist() == second["split"].tolist()
    train_ids = set(first.loc[first["split"].eq("train"), "source1_entity_id"])
    valid_ids = set(first.loc[first["split"].eq("valid"), "source1_entity_id"])
    assert train_ids.isdisjoint(valid_ids)
    assert train_ids | valid_ids == set(truth["source1_entity_id"])
    assert 0.1 < len(valid_ids) / len(truth) < 0.3


def test_create_split_and_streaming_recall(tmp_path: Path):
    truth = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "matched_entity_ids": ["S2-1,S3-1", "S2-2", ""],
        }
    )
    source1 = pd.DataFrame(
        {
            "entity_id": ["S1-1", "S1-2", "S1-3"],
            "country": ["IN", "US", "IN"],
        }
    )
    truth_path = tmp_path / "truth.tsv"
    source1_path = tmp_path / "source1.parquet"
    truth.to_csv(truth_path, sep="\t", index=False)
    source1.to_parquet(source1_path, index=False)

    manifest = create_split(truth_path, source1_path, tmp_path / "splits", 0.34, 9)
    assert manifest["train_rows"] + manifest["valid_rows"] == 3

    candidates = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-3"],
            "candidate_entity_id": ["S2-1", "S2-X", "S2-2", "S3-X"],
        }
    )
    candidates_path = tmp_path / "candidates.parquet"
    candidates.to_parquet(candidates_path, index=False)
    metrics = evaluate(candidates_path, truth_path, source1_path, batch_size=2)

    assert metrics["true_match_count"] == 3
    assert metrics["covered_match_count"] == 2
    assert metrics["match_recall"] == 2 / 3
    assert metrics["zero_candidate_count"] == 0
