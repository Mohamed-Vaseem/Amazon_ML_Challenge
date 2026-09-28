<#
.SYNOPSIS
    Amazon ML Challenge 2026 - Complete End-to-End Runner
    ONE SCRIPT: installs everything, runs every phase, saves final submission.

.DESCRIPTION
    Run this from the project root. It handles:
      Step 0  - Virtual environment + dependency installation
      Step 1  - Preprocessing (raw ZIP -> cleaned parquet) [SKIP if already done]
      Step 2  - Train/Validation split                     [SKIP if already done]
      Step 3  - Champion Phase 0: filter train source1     [SKIP if already done]
      Step 4  - Champion Phase 1: blocking candidates (24 rules, train)
      Step 5  - Champion Phase 2: rich feature engineering (train)
      Step 6  - Champion Phase 3: LightGBM ensemble (3 seeds x 5 folds)
      Step 7  - Champion Phase 4: blocking candidates (test)
      Step 8  - Champion Phase 5: rich feature engineering (test)
      Step 9  - Champion Phase 6: score test pairs
      Step 10 - Champion Phase 7: write final submission TSV

    FINAL OUTPUT:  output\champion\matching_results.tsv   <-- SUBMIT THIS

.PARAMETER RawZip
    Path to the challenge ZIP file downloaded from the portal.
    Default: C:\Users\$env:USERNAME\Downloads\6ab10eb3b23ba_student_resource.zip

.PARAMETER Threads
    Number of CPU threads for DuckDB. Default: all available cores (max 8).

.PARAMETER MemoryLimit
    DuckDB memory cap. Default: 10GB. Lower if you have <16 GB RAM.

.PARAMETER Force
    If set, re-runs ALL phases even if outputs already exist.

.EXAMPLE
    # First-time run from scratch (provide your ZIP path):
    .\RUN_FULL_PIPELINE.ps1 -RawZip "C:\Users\spadm\Downloads\6ab10eb3b23ba_student_resource.zip"

    # Resume from wherever it left off (no arguments needed):
    .\RUN_FULL_PIPELINE.ps1

    # Force rebuild everything:
    .\RUN_FULL_PIPELINE.ps1 -Force
#>

param(
    [string]$RawZip     = "C:\Users\$env:USERNAME\Downloads\6ab10eb3b23ba_student_resource.zip",
    [int]$Threads       = [math]::Min(8, [System.Environment]::ProcessorCount),
    [string]$MemoryLimit = "10GB",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$StartTime = Get-Date

# =======================================================================
#  HELPERS
# =======================================================================
function Log-Phase {
    param([string]$Msg)
    $ts = (Get-Date).ToString("HH:mm:ss")
    Write-Host ""
    Write-Host "[$ts] ==========================================" -ForegroundColor Cyan
    Write-Host "[$ts]  $Msg" -ForegroundColor Cyan
    Write-Host "[$ts] ==========================================" -ForegroundColor Cyan
}

function Log-Skip {
    param([string]$File)
    Write-Host "  [SKIP] - $File already exists" -ForegroundColor Green
}

function Log-Run {
    param([string]$Msg)
    Write-Host "  [RUNNING] - $Msg" -ForegroundColor Yellow
}

function Elapsed {
    $e = (Get-Date) - $StartTime
    return "$([math]::Floor($e.TotalHours))h $($e.Minutes)m $($e.Seconds)s"
}

function Need ([string]$Path) {
    # Returns $true if the file/dir is MISSING or --Force is set
    if ($Force) { return $true }
    return -not (Test-Path $Path)
}

# =======================================================================
#  STEP 0 - Virtual environment + install all dependencies
# =======================================================================
Log-Phase "STEP 0/10 - Environment Setup"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Log-Run "Creating virtual environment"
    python -m venv .venv
} else {
    Log-Skip ".venv"
}

Log-Run "Installing / verifying all dependencies"
.venv\Scripts\pip install --quiet --upgrade pip
.venv\Scripts\pip install --quiet -r requirements.txt
.venv\Scripts\pip install --quiet lightgbm>=4.3.0 scikit-learn>=1.4.0 duckdb>=0.10.0

# Confirm critical imports
.venv\Scripts\python -c @"
import lightgbm, sklearn, duckdb, rapidfuzz, pandas, pyarrow, numpy
print(f'  lightgbm {lightgbm.__version__}  sklearn {sklearn.__version__}')
print(f'  duckdb   {duckdb.__version__}  rapidfuzz {rapidfuzz.__version__}')
print('  All dependencies OK')
"@

# =======================================================================
#  STEP 1 - Preprocessing (raw ZIP -> cleaned parquet files)
# =======================================================================
Log-Phase "STEP 1/10 - Preprocessing (raw data -> cleaned parquet)"

$ProcessedDir = "data\processed"
$S1Train      = "$ProcessedDir\train_source1_clean.parquet"

if (Need $S1Train) {
    if (-not (Test-Path $RawZip)) {
        Write-Host ""
        Write-Host "  [WARN] Raw ZIP not found at: $RawZip" -ForegroundColor Red
        Write-Host "  [WARN] If data\processed\ already has the parquet files, this step will be skipped." -ForegroundColor Red
        Write-Host "  [WARN] Otherwise set -RawZip to the correct path." -ForegroundColor Red
    } else {
        Log-Run "Full preprocessing (this takes 30-90 min for large data)"
        .venv\Scripts\python -m src.preprocessing.run_preprocessing `
            --input          "$RawZip" `
            --output-dir     data `
            --mode           full `
            --chunk-size     250000
    }
} else {
    Log-Skip $S1Train
}

# Verify the 6 required parquet files exist before continuing
$RequiredParquets = @(
    "data\processed\train_source1_clean.parquet",
    "data\processed\train_source2_clean.parquet",
    "data\processed\train_source3_clean.parquet",
    "data\processed\test_source1_clean.parquet",
    "data\processed\test_source2_clean.parquet",
    "data\processed\test_source3_clean.parquet"
)
foreach ($p in $RequiredParquets) {
    if (-not (Test-Path $p)) {
        Write-Host "  [ERROR] MISSING: $p - cannot continue without preprocessed data." -ForegroundColor Red
        exit 1
    }
}
Write-Host "  [OK] All 6 source parquet files present" -ForegroundColor Green

# =======================================================================
#  STEP 2 - Train/Validation split (90% train, 10% holdout)
# =======================================================================
Log-Phase "STEP 2/10 - Train/Validation Split"

$TrainTruth = "data\splits\train_ground_truth.tsv"
$ValidTruth = "data\splits\valid_ground_truth.tsv"

if ((Need $TrainTruth) -or (Need $ValidTruth)) {
    # Find ground truth inside the ZIP/extracted folder
    $RawDir = Split-Path $RawZip -Parent
    $GTPath = "$RawDir\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_ground_truth.tsv"
    if (-not (Test-Path $GTPath)) {
        # Try common extraction paths
        $GTPath = Get-ChildItem -Path $RawDir -Recurse -Filter "train_ground_truth.tsv" `
                    -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
    }
    if ($GTPath -and (Test-Path $GTPath)) {
        Log-Run "Creating train/valid split (90/10, seed=2026)"
        .venv\Scripts\python -m src.pipeline.run_validation_pipeline `
            --ground-truth        "$GTPath" `
            --processed-dir       data\processed `
            --split-dir           data\splits `
            --candidates          data\candidates\valid_candidates.parquet `
            --candidate-pairs-tsv output\valid_candidate_pairs.tsv `
            --metrics             output\validation_metrics.json `
            --valid-fraction      0.10 `
            --seed                2026 `
            --batch-size          5000 `
            --top-k-per-source    80
    } else {
        Write-Host "  [WARN] Ground truth TSV not found - skipping split step." -ForegroundColor Yellow
        Write-Host "         Expected: $GTPath" -ForegroundColor Yellow
        Write-Host "         If data\splits\ already has the TSV files, this is fine." -ForegroundColor Yellow
    }
} else {
    Log-Skip "data\splits\ (train + valid ground truth TSVs)"
}

# Verify splits exist
foreach ($p in @($TrainTruth, $ValidTruth)) {
    if (-not (Test-Path $p)) {
        Write-Host "  [ERROR] MISSING: $p - cannot train without split files." -ForegroundColor Red
        exit 1
    }
}
Write-Host "  [OK] Train/valid split files present" -ForegroundColor Green

# =======================================================================
#  STEPS 3-10 - Champion Pipeline (all 7 phases, auto-skip completed)
# =======================================================================
Log-Phase "STEPS 3-10 - Champion Pipeline (Phases 0-7)"
Write-Host "  The pipeline checks each output file and skips phases already done." -ForegroundColor Gray
Write-Host "  Safe to re-run at any time - resumes from wherever it left off." -ForegroundColor Gray

# Build --force flag if requested
$ForceFlag = if ($Force) { "--force" } else { "" }

.venv\Scripts\python -m src.champion_model.run_champion_pipeline `
    --processed-dir        data/processed `
    --split-dir            data/splits `
    --work-dir             data/champion `
    --output-dir           output/champion `
    --model                models/champion_model.json `
    --threads              $Threads `
    --memory-limit         $MemoryLimit `
    --max-candidates       100 `
    --max-matches          11 `
    --max-train-rows       2000000 `
    --train-sample-modulus 6 `
    --n-seeds              3 `
    --n-folds              5 `
    --seed                 2026 `
    $(if ($Force) { "--force" })

# =======================================================================
#  DONE - Print summary
# =======================================================================
$Elapsed = Elapsed

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "        [OK] PIPELINE COMPLETE                            " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Total time    : $Elapsed" -ForegroundColor Green
Write-Host "  SUBMIT THIS   : output\champion\matching_results.tsv   " -ForegroundColor Green
Write-Host "  Candidates    : output\champion\candidate_pairs.tsv    " -ForegroundColor Green
Write-Host "  Val metrics   : output\champion\validation_metrics.json" -ForegroundColor Green
Write-Host "  Full report   : output\champion\final_report.json      " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

# Verify final output exists
if (Test-Path "output\champion\matching_results.tsv") {
    $Lines = (Get-Content "output\champion\matching_results.tsv" | Measure-Object -Line).Lines - 1
    Write-Host "  Submission rows : $Lines Source-1 entities" -ForegroundColor Green
} else {
    Write-Host "  [WARN] matching_results.tsv not found - check logs above for errors." -ForegroundColor Red
}
