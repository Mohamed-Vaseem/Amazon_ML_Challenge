"""
Run only the partitioned aggregation + ranking step on existing rule shards.

Usage:
    python scripts/run_aggregation.py [--work-dir data/advanced_hybrid] [--force]
"""
import argparse
import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.advanced_hybrid.blocking import (
    _partitioned_aggregate_and_rank,
    validate_candidate_schema,
    sql_path,
)


def main():
    parser = argparse.ArgumentParser(description="Run partitioned aggregation only")
    parser.add_argument("--work-dir", default="data/advanced_hybrid")
    parser.add_argument("--source1", default="data/advanced_hybrid/model_train_source1.parquet")
    parser.add_argument("--output", default="data/advanced_hybrid/train_candidates.parquet")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--max-candidates", type=int, default=120)
    parser.add_argument("--num-partitions", type=int, default=16)
    parser.add_argument("--force", action="store_true",
                        help="Ignore cached partition files and recompute")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    source1 = Path(args.source1)
    output = Path(args.output)
    temp_dir = work_dir / "tmp_train_cands"

    assert source1.exists(), f"source1 not found: {source1}"
    assert temp_dir.exists(), f"temp_dir not found: {temp_dir}"

    # Check rule shards exist
    shards = sorted(temp_dir.glob("rule_*.parquet"))
    print(f"Found {len(shards)} rule shards in {temp_dir}")
    if len(shards) == 0:
        print("ERROR: No rule shards found. Run full pipeline Phase 1 first.")
        sys.exit(1)

    shards_glob = sql_path(temp_dir / "rule_*.parquet")

    if args.force:
        # Clear partitions dir
        part_dir = temp_dir / "partitions"
        if part_dir.exists():
            for f in part_dir.glob("*.parquet"):
                f.unlink()
            print(f"Cleared {part_dir}")
        # Clear s1 map
        s1_map = temp_dir / "s1_row_map.parquet"
        if s1_map.exists():
            s1_map.unlink()
            print("Cleared s1_row_map.parquet")
        # Clear any existing output
        if output.exists():
            output.unlink()
            print(f"Cleared {output}")

    print(f"\nRunning partitioned aggregation: {args.num_partitions} partitions, "
          f"{args.threads} threads, memory_limit={args.memory_limit}")
    print(f"  shards_glob: {shards_glob}")
    print(f"  output: {output}\n")

    _partitioned_aggregate_and_rank(
        source1=source1,
        shards_glob=shards_glob,
        output_path=output,
        temp_dir=temp_dir,
        num_partitions=args.num_partitions,
        max_candidates=args.max_candidates,
        threads=args.threads,
        memory_limit=args.memory_limit,
    )

    print("\nValidating candidate schema...")
    validate_candidate_schema(output)
    print(f"SUCCESS: candidates written and validated -> {output}")
    print(f"  file size: {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
