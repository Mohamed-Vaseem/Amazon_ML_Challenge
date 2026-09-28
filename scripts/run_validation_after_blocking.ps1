param(
    [Parameter(Mandatory = $true)]
    [int]$BlockingProcessId,
    [Parameter(Mandatory = $true)]
    [string]$GroundTruth
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$expectedCandidates = Join-Path $projectRoot "data\candidates\test_candidates.parquet"
$expectedPairs = Join-Path $projectRoot "output\candidate_pairs.tsv"

Set-Location $projectRoot
$process = Get-Process -Id $BlockingProcessId -ErrorAction SilentlyContinue
if ($null -ne $process) {
    Write-Output "Waiting for blocking process $BlockingProcessId..."
    Wait-Process -Id $BlockingProcessId
}

if (-not (Test-Path -LiteralPath $expectedCandidates) -or (Get-Item -LiteralPath $expectedCandidates).Length -eq 0) {
    throw "Blocking process ended without a non-empty $expectedCandidates"
}
if (-not (Test-Path -LiteralPath $expectedPairs) -or (Get-Item -LiteralPath $expectedPairs).Length -eq 0) {
    throw "Blocking process ended without a non-empty $expectedPairs"
}

& $python -m src.pipeline.run_validation_pipeline `
    --ground-truth $GroundTruth `
    --processed-dir data/processed `
    --split-dir data/splits `
    --candidates data/candidates/valid_candidates.parquet `
    --candidate-pairs-tsv output/valid_candidate_pairs.tsv `
    --metrics output/validation_metrics.json `
    --valid-fraction 0.10 `
    --seed 2026 `
    --batch-size 5000 `
    --top-k-per-source 80 `
    --prefilter-limit-per-source 1200 `
    --min-match-recall 0.95

if ($LASTEXITCODE -ne 0) {
    throw "Validation pipeline failed with exit code $LASTEXITCODE"
}
