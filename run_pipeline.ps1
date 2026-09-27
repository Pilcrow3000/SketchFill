# run_pipeline.ps1 — SketchFill Training Pipeline
# Runs: data download → training → density filter
# Launch app separately with: .\run_app.ps1 or streamlit run app.py
# Prerequisites: run .\setup_venv.ps1 first
# Usage: .\run_pipeline.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$python = ".\venv\Scripts\python.exe"

# Verify venv exists
if (-not (Test-Path $python)) {
    Write-Error "Virtual environment not found. Run .\setup_venv.ps1 first."
    exit 1
}

Write-Host ""
Write-Host "╔════════════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "║       SketchFill — Training Pipeline           ║" -ForegroundColor Magenta
Write-Host "╚════════════════════════════════════════════════╝" -ForegroundColor Magenta
Write-Host ""

# ── Step 1: Download & Preprocess Data ───────────────────────────────────────
Write-Host "══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Step 1/3 — Download & Preprocess Data"           -ForegroundColor Cyan
Write-Host "══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Downloads 6 QuickDraw categories (~1.5 GB)"
Write-Host "  Resizes from 28x28 to 128x128 (bicubic)"
Write-Host ""
& $python data/download_data.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Data download failed (exit code $LASTEXITCODE)"
    exit $LASTEXITCODE
}
Write-Host ""
Write-Host "  ✓ Data ready" -ForegroundColor Green
Write-Host ""

# ── Step 2: Train CVAE ────────────────────────────────────────────────────────
Write-Host "══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Step 2/3 — Train CVAE"                           -ForegroundColor Cyan
Write-Host "══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Training on RTX 3070 Ti — estimated 30-60 min"
Write-Host "  Outputs: best_model.pt, loss_curves.png,"
Write-Host "           sample_reconstructions.png, training_summary.json"
Write-Host ""
& $python models/train.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Training failed (exit code $LASTEXITCODE)"
    exit $LASTEXITCODE
}
Write-Host ""
Write-Host "  ✓ Training complete — check checkpoints/ for results" -ForegroundColor Green
Write-Host ""

# ── Step 3: Fit Density Filter ───────────────────────────────────────────────
Write-Host "══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Step 3/3 — Fit Density Filter"                   -ForegroundColor Cyan
Write-Host "══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Extracting latent codes + fitting KDE per class"
Write-Host "  Outputs: density_model.pkl, latent_codes.pkl,"
Write-Host "           density_filter_test.png"
Write-Host ""
& $python models/density_filter.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Density filter fitting failed (exit code $LASTEXITCODE)"
    exit $LASTEXITCODE
}
Write-Host ""
Write-Host "  ✓ Density model ready" -ForegroundColor Green
Write-Host ""

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host "══════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  Pipeline Complete!"                               -ForegroundColor Green
Write-Host "══════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "All artifacts saved to: checkpoints/" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review training results in checkpoints/" -ForegroundColor White
Write-Host "  2. Launch the app: .\run_app.ps1" -ForegroundColor White
Write-Host "     or manually: streamlit run app.py" -ForegroundColor White
Write-Host ""
