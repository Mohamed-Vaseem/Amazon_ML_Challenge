import pyarrow.parquet as pq
from pathlib import Path
import json
import shutil
import time

def finalize():
    t0 = time.perf_counter()
    preds_parquet = Path('data/advanced_hybrid/test_predictions.unsorted.parquet')
    source1_parquet = Path('data/processed/test_source1_clean.parquet')
    
    # We may need to get threshold from metadata
    meta = json.loads(Path('models/advanced_hybrid/model_metadata.json').read_text())
    threshold = meta['threshold']
    max_matches = 11

    print(f"Loading predictions...")
    preds_by_s1 = {}
    pf = pq.ParquetFile(preds_parquet)
    for batch in pf.iter_batches(batch_size=500_000):
        d = batch.to_pydict()
        for s1, c, sc in zip(d["source1_entity_id"], d["candidate_entity_id"], d["match_score"]):
            preds_by_s1.setdefault(str(s1), []).append((str(c), float(sc)))

    # Sort each list of predictions by score DESC
    for pairs in preds_by_s1.values():
        pairs.sort(key=lambda x: x[1], reverse=True)

    print("Writing result.tsv...")
    s1_pf = pq.ParquetFile(source1_parquet)
    
    matching_tsv = Path('result.tsv')
    with matching_tsv.open("w", encoding="utf-8", newline="") as mf:
        mf.write("source1_entity_id\tmatched_entity_ids\n")

        for batch in s1_pf.iter_batches(batch_size=100_000, columns=["entity_id"]):
            entity_ids = batch.column("entity_id").to_pylist()
            for s1_id in entity_ids:
                pairs = preds_by_s1.get(s1_id, [])

                seen_matches = set()
                match_list = []
                for c, sc in pairs:
                    if sc >= threshold:
                        if c not in seen_matches:
                            seen_matches.add(c)
                            match_list.append(c)
                            if len(match_list) >= max_matches:
                                break
                
                mf.write(f"{s1_id}\t{','.join(match_list)}\n")

    print(f"Done in {time.perf_counter()-t0:.1f} sec!")

if __name__ == "__main__":
    finalize()
