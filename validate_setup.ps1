# validate_setup.ps1 - Pre-flight check for SketchFill
# Validates environment and dependencies before running pipeline
# Usage: .\validate_setup.ps1

Set-StrictMode -Version Latest

$ErrorActionPreference = "Continue"

$python = ".\venv\Scripts\python.exe"

$issues = 0

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  SketchFill - Pre-Flight Validation" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check 1: Virtual environment
Write-Host "[1/7] Checking virtual environment..." -ForegroundColor Yellow

if (Test-Path $python) {
    Write-Host "  [OK] venv found" -ForegroundColor Green
}
else {
    Write-Host "  [FAIL] venv not found - run .\setup_venv.ps1" -ForegroundColor Red
    $issues++
}

# Check 2: PyTorch + CUDA
Write-Host "[2/7] Checking PyTorch + CUDA..." -ForegroundColor Yellow

if (Test-Path $python) {
    $torchCheck = & $python -c "import torch; print(torch.__version__, torch.cuda.is_available())" 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] PyTorch installed: $torchCheck" -ForegroundColor Green

        if ($torchCheck -notmatch "True") {
            Write-Host "  [WARN] CUDA not available - training will be slow on CPU" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "  [FAIL] PyTorch not installed" -ForegroundColor Red
        $issues++
    }
}

# Check 3: Required packages
Write-Host "[3/7] Checking required packages..." -ForegroundColor Yellow

$required = @(
    "numpy",
    "scikit-learn",
    "streamlit",
    "plotly",
    "pillow",
    "tqdm",
    "matplotlib",
    "pandas"
)

$missing = @()

if (Test-Path $python) {
    foreach ($pkg in $required) {

        # Map pip package names to Python import names
        $importName = switch ($pkg) {
            "scikit-learn" { "sklearn" }
            "pillow"      { "PIL" }
            default       { $pkg }
        }

        $check = & $python -c "import $importName" 2>&1

        if ($LASTEXITCODE -ne 0) {
            $missing += $pkg
        }
    }

    if ($missing.Count -eq 0) {
        Write-Host "  [OK] All packages installed" -ForegroundColor Green
    }
    else {
        Write-Host "  [FAIL] Missing packages: $($missing -join ', ')" -ForegroundColor Red
        Write-Host "  [INFO] Run: pip install -r requirements.txt" -ForegroundColor Yellow
        $issues++
    }
}

# Check 4: Directory structure
Write-Host "[4/7] Checking directory structure..." -ForegroundColor Yellow

$dirs = @(
    "data\raw",
    "data\processed",
    "models",
    "checkpoints"
)

$missing_dirs = @()

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        $missing_dirs += $dir
    }
}

if ($missing_dirs.Count -eq 0) {
    Write-Host "  [OK] All directories present" -ForegroundColor Green
}
else {
    Write-Host "  [FAIL] Missing directories: $($missing_dirs -join ', ')" -ForegroundColor Red
    $issues++
}

# Check 5: Core Python files
Write-Host "[5/7] Checking core Python files..." -ForegroundColor Yellow

$files = @(
    "data\download_data.py",
    "models\cvae_model.py",
    "models\train.py",
    "models\density_filter.py",
    "app.py"
)

$missing_files = @()

foreach ($file in $files) {
    if (-not (Test-Path $file)) {
        $missing_files += $file
    }
}

if ($missing_files.Count -eq 0) {
    Write-Host "  [OK] All core files present" -ForegroundColor Green
}
else {
    Write-Host "  [FAIL] Missing files: $($missing_files -join ', ')" -ForegroundColor Red
    $issues++
}

# Check 6: Syntax validation
Write-Host "[6/7] Validating Python syntax..." -ForegroundColor Yellow

if (Test-Path $python) {

    $syntax_errors = 0

    foreach ($file in $files) {

        if (Test-Path $file) {

            $result = & $python -m py_compile $file 2>&1

            if ($LASTEXITCODE -ne 0) {
                Write-Host "  [FAIL] Syntax error in $file" -ForegroundColor Red
                $syntax_errors++
            }
        }
    }

    if ($syntax_errors -eq 0) {
        Write-Host "  [OK] All Python files valid" -ForegroundColor Green
    }
    else {
        $issues += $syntax_errors
    }
}

# Check 7: Disk space
Write-Host "[7/7] Checking disk space..." -ForegroundColor Yellow

$drive = (Get-Location).Drive.Name
$disk = Get-PSDrive $drive
$free_gb = [math]::Round($disk.Free / 1GB, 2)

if ($free_gb -gt 5) {
    Write-Host "  [OK] Sufficient disk space: $free_gb GB free" -ForegroundColor Green
}
else {
    Write-Host "  [WARN] Low disk space: $free_gb GB free (recommend 5+ GB)" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan

if ($issues -eq 0) {

    Write-Host "  [OK] All checks passed - ready to run pipeline!" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""

    Write-Host "Next step: .\run_pipeline.ps1" -ForegroundColor White
}
else {

    Write-Host "  [FAIL] Found $issues issue(s) - fix before running" -ForegroundColor Red
    Write-Host "============================================" -ForegroundColor Cyan

    exit 1
}

Write-Host ""