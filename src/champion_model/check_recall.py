"""Quick diagnostic: check how many candidates the new blocking rules add.

Run after train candidates are built:
  python -m src.champion_model.check_recall

Prints candidate recall vs the baseline blocking recall of 77.7%.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


def main():
    valid_truth = Path("data/splits/valid_ground_truth.tsv")
    champion_cands = Path("data/champion/train_candidates.parquet")
    baseline_cands = Path("data/advanced_fast/train_candidates.parquet")

    if not champion_cands.is_file():
        print("Champion candidates not yet built — run Phase 1 first.")
        return

    con = duckdb.connect()

    def recall_for(cands_path: Path, label: str):
        if not cands_path.is_file():
            print(f"  {label}: file missing")
            return
        result = con.execute(f"""
        WITH truth AS (
          SELECT source1_entity_id, trim(unnest(string_split(matched_entity_ids, ','))) AS match_id
          FROM read_csv('{str(valid_truth).replace(chr(92),'/')}',
                        delim='\\t', header=true, all_varchar=true)
          WHERE matched_entity_ids <> ''
        ),
        cands AS (
          SELECT source1_entity_id, candidate_entity_id
          FROM read_parquet('{str(cands_path).replace(chr(92),'/')}')
        ),
        hits AS (
          SELECT t.source1_entity_id, t.match_id
          FROM truth t
          JOIN cands c ON t.source1_entity_id = c.source1_entity_id
                       AND t.match_id = c.candidate_entity_id
        )
        SELECT
          count(*) AS total_truth,
          (SELECT count(*) FROM hits) AS covered,
          round(100.0 * (SELECT count(*) FROM hits) / count(*), 2) AS recall_pct
        FROM truth
        """).fetchone()
        print(f"  {label}: recall = {result[2]}%  ({result[1]}/{result[0]} truth pairs covered)")

    print("=== Candidate Recall Comparison ===")
    recall_for(baseline_cands, "Baseline (13 rules, max_cands=64) ")
    recall_for(champion_cands, "Champion (24 rules, max_cands=100)")
    con.close()


if __name__ == "__main__":
    main()
