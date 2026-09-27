# SketchFill - Launch Web Application
# Starts the Streamlit interface for interactive sketch generation

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SketchFill - Launching Application" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run setup_venv.ps1 first" -ForegroundColor Yellow
    exit 1
}

# Check if model checkpoint exists
if (-not (Test-Path "checkpoints\best_model.pt")) {
    Write-Host "WARNING: Model checkpoint not found!" -ForegroundColor Yellow
    Write-Host "Please train the model first: python models\train.py" -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Continue anyway? (y/n)"
    if ($response -ne "y") {
        exit 1
    }
}

# Activate environment and run
Write-Host "Starting Streamlit application..." -ForegroundColor Green
Write-Host "App will open in your browser at http://localhost:8501" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

& .\venv\Scripts\Activate.ps1
streamlit run app.py
