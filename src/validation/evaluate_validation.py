"""Stream validation candidates and report detailed blocking recall."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PAIR_SEPARATOR = "\x1f"


def parse_truth_list(value: object) -> list[str]:
    return [part for part in str(value or "").split(",") if part]


def evaluate(
    candidates_path: Path,
    ground_truth_path: Path,
    source1_path: Path,
    batch_size: int = 500_000,
) -> dict[str, object]:
    truth = pd.read_csv(ground_truth_path, sep="\t", dtype=str, keep_default_na=False)
    required = {"source1_entity_id", "matched_entity_ids"}
    if not required.issubset(truth.columns):
        raise ValueError(f"Ground truth must contain {sorted(required)}")

    source1 = pd.read_parquet(source1_path, columns=["entity_id", "country"]).fillna("")
    country_by_id = dict(zip(source1["entity_id"].astype(str), source1["country"].astype(str)))

    truth_by_entity: dict[str, set[str]] = {}
    truth_pair_keys: set[str] = set()
    truth_by_source: Counter[str] = Counter()
    truth_by_country: Counter[str] = Counter()
    for row in truth.itertuples(index=False):
        source1_id = str(row.source1_entity_id)
        matches = set(parse_truth_list(row.matched_entity_ids))
        truth_by_entity[source1_id] = matches
        country = country_by_id.get(source1_id, "")
        for candidate_id in matches:
            truth_pair_keys.add(f"{source1_id}{PAIR_SEPARATOR}{candidate_id}")
            truth_by_source[candidate_id.split("-", 1)[0]] += 1
            truth_by_country[country] += 1

    covered_pair_keys: set[str] = set()
    candidate_counts: Counter[str] = Counter()
    candidate_rows = 0
    parquet = pq.ParquetFile(candidates_path)
    columns = ["source1_entity_id", "candidate_entity_id"]
    for batch_no, batch in enumerate(parquet.iter_batches(batch_size=batch_size, columns=columns), start=1):
        frame = batch.to_pandas().fillna("")
        source1_ids = frame["source1_entity_id"].astype(str)
        candidate_ids = frame["candidate_entity_id"].astype(str)
        candidate_rows += len(frame)
        candidate_counts.update(source1_ids.value_counts().to_dict())
        keys = source1_ids + PAIR_SEPARATOR + candidate_ids
        hits = keys.isin(truth_pair_keys)
        if hits.any():
            covered_pair_keys.update(keys[hits].tolist())
        if batch_no % 20 == 0:
            print(f"evaluated {candidate_rows:,} candidate rows", flush=True)

    entity_ids = truth["source1_entity_id"].astype(str).tolist()
    matched_entity_ids = [entity_id for entity_id in entity_ids if truth_by_entity[entity_id]]
    singleton_ids = [entity_id for entity_id in entity_ids if not truth_by_entity[entity_id]]
    covered_by_entity: defaultdict[str, int] = defaultdict(int)
    covered_by_source: Counter[str] = Counter()
    covered_by_country: Counter[str] = Counter()
    for key in covered_pair_keys:
        source1_id, candidate_id = key.split(PAIR_SEPARATOR, 1)
        covered_by_entity[source1_id] += 1
        covered_by_source[candidate_id.split("-", 1)[0]] += 1
        covered_by_country[country_by_id.get(source1_id, "")] += 1

    full_recall_count = sum(
        covered_by_entity[entity_id] == len(truth_by_entity[entity_id]) for entity_id in matched_entity_ids
    )
    any_recall_count = sum(covered_by_entity[entity_id] > 0 for entity_id in matched_entity_ids)
    zero_candidate_count = sum(candidate_counts[entity_id] == 0 for entity_id in entity_ids)
    counts = np.fromiter((candidate_counts[entity_id] for entity_id in entity_ids), dtype=np.int64)

    source_recall = {
        source: {
            "true_matches": int(total),
            "covered_matches": int(covered_by_source[source]),
            "recall": covered_by_source[source] / total if total else math.nan,
        }
        for source, total in sorted(truth_by_source.items())
    }
    country_recall = {
        country: {
            "true_matches": int(total),
            "covered_matches": int(covered_by_country[country]),
            "recall": covered_by_country[country] / total if total else math.nan,
        }
        for country, total in sorted(truth_by_country.items())
    }

    true_match_count = len(truth_pair_keys)
    matched_entity_count = len(matched_entity_ids)
    return {
        "entity_count": len(entity_ids),
        "matched_entity_count": matched_entity_count,
        "singleton_count": len(singleton_ids),
        "true_match_count": true_match_count,
        "covered_match_count": len(covered_pair_keys),
        "match_recall": len(covered_pair_keys) / true_match_count if true_match_count else math.nan,
        "matched_entity_any_recall": any_recall_count / matched_entity_count if matched_entity_count else math.nan,
        "matched_entity_full_recall": full_recall_count / matched_entity_count if matched_entity_count else math.nan,
        "candidate_rows": candidate_rows,
        "mean_candidates_per_entity": float(counts.mean()) if len(counts) else 0.0,
        "median_candidates_per_entity": float(np.median(counts)) if len(counts) else 0.0,
        "p95_candidates_per_entity": float(np.percentile(counts, 95)) if len(counts) else 0.0,
        "max_candidates_per_entity": int(counts.max()) if len(counts) else 0,
        "zero_candidate_count": zero_candidate_count,
        "zero_candidate_rate": zero_candidate_count / len(entity_ids) if entity_ids else math.nan,
        "recall_by_candidate_source": source_recall,
        "recall_by_country": country_recall,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate candidate recall without loading all candidates into RAM.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--source1", required=True)
    parser.add_argument("--output", default="output/validation_metrics.json")
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument("--min-match-recall", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(
        Path(args.candidates),
        Path(args.ground_truth),
        Path(args.source1),
        args.batch_size,
    )
    metrics["target_match_recall"] = args.min_match_recall
    metrics["target_passed"] = bool(metrics["match_recall"] >= args.min_match_recall)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, allow_nan=False), flush=True)
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
