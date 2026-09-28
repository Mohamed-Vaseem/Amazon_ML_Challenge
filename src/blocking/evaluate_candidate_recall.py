"""Evaluate blocking recall against train_ground_truth.tsv."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.blocking.advanced_blocking import score_candidate_recall


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure candidate recall on training/validation candidates.")
    parser.add_argument("--candidates", required=True, help="Long-format candidate parquet.")
    parser.add_argument("--ground-truth", required=True, help="train_ground_truth.tsv or validation subset TSV.")
    parser.add_argument("--country-breakdown-source1", help="Optional cleaned source1 parquet for country breakdown.")
    args = parser.parse_args()

    candidates = pd.read_parquet(args.candidates, columns=["source1_entity_id", "candidate_entity_id"])
    truth = pd.read_csv(args.ground_truth, sep="\t", dtype=str, keep_default_na=False)
    metrics = score_candidate_recall(candidates, truth)

    if args.country_breakdown_source1:
        source1 = pd.read_parquet(args.country_breakdown_source1, columns=["entity_id", "country"])
        truth_with_country = truth.merge(source1, left_on="source1_entity_id", right_on="entity_id", how="left")
        by_country = {}
        for country, group in truth_with_country.groupby("country", dropna=False):
            by_country[str(country)] = score_candidate_recall(candidates, group[["source1_entity_id", "matched_entity_ids"]])
        metrics["by_country"] = by_country

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
