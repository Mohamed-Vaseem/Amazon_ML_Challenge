import pandas as pd

from src.blocking.advanced_blocking import BlockingConfig, MultiSourceBlocker, score_candidate_recall


def frame(rows):
    return pd.DataFrame(rows)


def test_blocking_finds_multilingual_and_postal_candidates():
    s1 = frame(
        [
            {
                "entity_id": "S1-1",
                "country": "India",
                "name_core": "राम मार्केटिंग",
                "address_clean": "new delhi 110001",
                "important_name_tokens": "राम|मार्केटिंग",
                "numbers": "110001",
                "postal_code": "110001",
            }
        ]
    )
    s2 = frame(
        [
            {
                "entity_id": "S2-1",
                "country": "India",
                "name_core": "राम मार्केटिंग",
                "address_clean": "delhi 110001",
                "important_name_tokens": "राम|मार्केटिंग",
                "numbers": "110001",
                "postal_code": "110001",
            },
            {
                "entity_id": "S2-2",
                "country": "US",
                "name_core": "ram marketing",
                "address_clean": "new york 10001",
                "important_name_tokens": "ram|marketing",
                "numbers": "10001",
                "postal_code": "10001",
            },
        ]
    )
    s3 = frame(
        [
            {
                "entity_id": "S3-1",
                "country": "India",
                "name_core": "other business",
                "address_clean": "mumbai 400001",
                "important_name_tokens": "other|business",
                "numbers": "400001",
                "postal_code": "400001",
            }
        ]
    )

    blocker = MultiSourceBlocker(
        s2,
        s3,
        BlockingConfig(top_k_per_source=5, min_token_df=1, max_token_df_ratio=1.0),
    )
    candidates = blocker.generate_chunk(s1)

    assert "S2-1" in set(candidates["candidate_entity_id"])
    assert "S2-2" not in set(candidates["candidate_entity_id"])


def test_candidate_recall_metric():
    candidates = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-1", "S1-2"],
            "candidate_entity_id": ["S2-1", "S3-9", "S2-8"],
        }
    )
    truth = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "matched_entity_ids": ["S2-1,S3-1", "S2-8", ""],
        }
    )

    metrics = score_candidate_recall(candidates, truth)

    assert metrics["true_match_count"] == 3
    assert metrics["match_recall"] == 2 / 3
    assert metrics["singleton_count"] == 1
