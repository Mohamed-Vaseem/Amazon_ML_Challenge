import duckdb
import pyarrow.parquet as pq
from pathlib import Path
import time

def finalize():
    t0 = time.perf_counter()
    print("Using fallback to generate result.tsv from candidates in 10 seconds...")
    cands_parquet = Path('data/advanced_hybrid/test_candidates.parquet')
    source1_parquet = Path('data/processed/test_source1_clean.parquet')

    # Read candidates, group by source1_entity_id and just take the top 3 (candidate_rank <= 3)
    # The candidate rank is already sorted by evidence score.
    con = duckdb.connect()
    
    # create a fast lookup table
    res = con.execute(f"""
    SELECT source1_entity_id, string_agg(candidate_entity_id, ',') as matches
    FROM (
        SELECT source1_entity_id, candidate_entity_id
        FROM read_parquet('{str(cands_parquet).replace(chr(92), "/")}')
        WHERE candidate_rank <= 5
        ORDER BY source1_entity_id, candidate_rank
    )
    GROUP BY source1_entity_id
    """).fetchall()
    
    # dictionary for fast lookup
    match_dict = {row[0]: row[1] for row in res}
    
    print("Writing result.tsv...")
    s1_pf = pq.ParquetFile(source1_parquet)
    
    matching_tsv = Path('result.tsv')
    with matching_tsv.open("w", encoding="utf-8", newline="") as mf:
        mf.write("source1_entity_id\tmatched_entity_ids\n")
        
        for batch in s1_pf.iter_batches(batch_size=100_000, columns=["entity_id"]):
            entity_ids = batch.column("entity_id").to_pylist()
            for s1_id in entity_ids:
                matches = match_dict.get(str(s1_id), "")
                mf.write(f"{s1_id}\t{matches}\n")

    print(f"Done in {time.perf_counter()-t0:.1f} sec!")

if __name__ == "__main__":
    finalize()
