# Amazon ML Challenge 2026 — Business Entity Resolution

This repository contains the complete preprocessing and modeling pipeline for the Amazon ML Challenge 2026 Business Entity Resolution task. 

## 1. What Is This Challenge?

**Task:** Given three sets of business-entity records from different sources (Source 1, Source 2, Source 3), find which Source-1 entity matches which Source-2 / Source-3 entities.

**Metric:** Macro-averaged **F0.5** (precision-weighted F-score) per Source-1 entity.  
Formula: `F0.5 = 1.25 × (precision × recall) / (0.25 × precision + recall)`

## 2. Project Structure

```text
AMAZON_ML_CHALLENGE/
│
├── src/                          ← All Python source code
│   ├── preprocessing/            ← Cleans raw TSVs → parquet
│   ├── blocking/                 ← Python-based blocker
│   ├── features/                 ← Pair feature builder
│   ├── modeling/                 ← Linear pair scorer
│   ├── pipeline/                 ← Integrated validation pipeline
│   ├── champion_model/           ← 🏆 The main Champion pipeline
│   │   ├── blocking.py           ←   24-rule DuckDB SQL blocker
│   │   ├── features.py           ←   60+ feature engineering
│   │   ├── model.py              ←   LightGBM ensemble trainer
│   │   ├── run_champion_pipeline.py  ← End-to-end runner
│   │   └── check_recall.py       ←   Recall diagnostic tool
│   └── validation/               ← Recall evaluation helpers
│
├── data/                         ← Ignored in Git (Data storage)
├── models/                       ← Ignored in Git (Model weights)
├── output/                       ← Ignored in Git (Submission files & logs)
│
├── requirements.txt              ← Core dependencies
├── requirements_champion.txt     ← Champion model dependencies (LightGBM etc.)
└── RUN_FULL_PIPELINE.ps1         ← 🏆 One-click pipeline runner
```

## 3. Environment Setup

```powershell
# Clone/pull the repo, then:
python -m venv .venv
.venv\Scripts\Activate.ps1

# Core dependencies (preprocessing, blocking, features)
pip install -r requirements.txt

# Champion model dependencies (LightGBM, scikit-learn)
pip install -r requirements_champion.txt
```

## 4. How to Run the Pipeline

The entire process is automated through a single PowerShell script. This handles preprocessing, training, validation, and generating the final submission file. 

```powershell
# Replace <YOUR_USERNAME> with your Windows username
.\RUN_FULL_PIPELINE.ps1 -RawZip "C:\Users\<YOUR_USERNAME>\Downloads\6ab10eb3b23ba_student_resource.zip"
```
*(The script is idempotent. If it fails or stops, rerunning it will safely pick up from where it left off).*

## 5. Pipeline Architecture 

1. **Preprocessing:** Normalizes text, extracts features (emails, postal codes, URLs), and stores the data as `.parquet` files.
2. **Blocking (DuckDB):** Uses 24 SQL-based rules to generate candidate pairs (targets ≥ 92% recall ceiling).
3. **Feature Engineering:** Extracts 60+ features per candidate pair (Jaro-Winkler, Levenshtein, Jaccard, Rank-context).
4. **LightGBM Ensemble:** Trains 15 LightGBM models (3 random seeds × 5 folds) with a hard-positive override.
5. **Scoring & Submission:** Tuned threshold maximizes macro-F0.5. Results are written to `output/champion/matching_results.tsv`.

## 6. Handoff Schema (Core Columns)
- `entity_id`
- `country_clean`
- `name_core`
- `name_ascii`
- `address_clean`
- `address_ascii`
- `important_name_tokens`
- `numbers`
- `house_number`
- `postal_code`
- `emails`, `phones`, `urls`
- `record_fingerprint`

## 7. Submission Format

The final generated submission will be at `output/champion/matching_results.tsv` and follows this format:
```tsv
source1_entity_id	matched_entity_ids
S1-0000001	S2-1234567,S3-9876543
S1-0000002	
S1-0000003	S2-5555555
```

## Note

- Never commit data (`.parquet`, `.tsv`) or model files (`.json`) to Git.
- Always run the script sequentially and monitor output logs.
