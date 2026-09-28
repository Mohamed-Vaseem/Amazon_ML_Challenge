"""Run validation split, blocking, and streaming recall evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the post-blocking validation pipeline.")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--candidates", default="data/candidates/valid_candidates.parquet")
    parser.add_argument("--candidate-pairs-tsv", default="output/valid_candidate_pairs.tsv")
    parser.add_argument("--metrics", default="output/validation_metrics.json")
    parser.add_argument("--valid-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--top-k-per-source", type=int, default=80)
    parser.add_argument("--prefilter-limit-per-source", type=int, default=1200)
    parser.add_argument("--min-match-recall", type=float, default=0.95)
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-blocking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed = Path(args.processed_dir)
    split_dir = Path(args.split_dir)
    source1 = processed / "train_source1_clean.parquet"
    source2 = processed / "train_source2_clean.parquet"
    source3 = processed / "train_source3_clean.parquet"
    valid_truth = split_dir / "valid_ground_truth.tsv"
    valid_ids = split_dir / "valid_source1_ids.txt"

    if not args.skip_split:
        run(
            [
                sys.executable,
                "-m",
                "src.validation.create_validation_split",
                "--ground-truth",
                args.ground_truth,
                "--source1",
                str(source1),
                "--output-dir",
                str(split_dir),
                "--valid-fraction",
                str(args.valid_fraction),
                "--seed",
                str(args.seed),
            ]
        )

    if not args.skip_blocking:
        run(
            [
                sys.executable,
                "-m",
                "src.blocking.run_blocking",
                "--source1",
                str(source1),
                "--source2",
                str(source2),
                "--source3",
                str(source3),
                "--s1-id-filter",
                str(valid_ids),
                "--output",
                args.candidates,
                "--candidate-pairs-tsv",
                args.candidate_pairs_tsv,
                "--batch-size",
                str(args.batch_size),
                "--top-k-per-source",
                str(args.top_k_per_source),
                "--prefilter-limit-per-source",
                str(args.prefilter_limit_per_source),
            ]
        )

    run(
        [
            sys.executable,
            "-m",
            "src.validation.evaluate_validation",
            "--candidates",
            args.candidates,
            "--ground-truth",
            str(valid_truth),
            "--source1",
            str(source1),
            "--output",
            args.metrics,
            "--min-match-recall",
            str(args.min_match_recall),
        ]
    )


if __name__ == "__main__":
    main()
