# setup_venv.ps1 � SketchFill environment bootstrap
# Target: Python 3.11, PyTorch 2.2.2 + CUDA 12.1, RTX 3070 Ti
# Run from project root: .\setup_venv.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PythonVersion = "3.11"
$VenvDir = "venv"

Write-Host "=== SketchFill Environment Setup ===" -ForegroundColor Magenta

# Check Python
Write-Host "==> Checking Python $PythonVersion..." -ForegroundColor Cyan
try {
    $ver = py -$PythonVersion --version 2>&1
    Write-Host "    Found: $ver" -ForegroundColor Green
} catch {
    Write-Error "Python $PythonVersion not found. Get it from https://python.org"
    exit 1
}

# Create venv
if (Test-Path $VenvDir) {
    Write-Host "==> venv already exists, skipping." -ForegroundColor Yellow
} else {
    Write-Host "==> Creating venv..." -ForegroundColor Cyan
    py -$PythonVersion -m venv $VenvDir
}

$pip    = ".\$VenvDir\Scripts\pip.exe"
$python = ".\$VenvDir\Scripts\python.exe"

# Upgrade pip and install uv
Write-Host "==> Upgrading pip and installing uv..." -ForegroundColor Cyan
& $python -m pip install --upgrade pip uv --quiet

# PyTorch CUDA (using uv - much faster)
Write-Host "==> Installing PyTorch 2.2.2 + CUDA 12.1 with uv (~2 GB)..." -ForegroundColor Cyan
& $python -m uv pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Other deps (using uv - much faster, no NVIDIA index warnings)
Write-Host "==> Installing project dependencies with uv..." -ForegroundColor Cyan
& $python -m uv pip install -r requirements.txt

# Smoke test
Write-Host "==> CUDA smoke test..." -ForegroundColor Cyan
& $python -c @"
import torch
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU     : {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'VRAM    : {props.total_memory/1024**3:.1f} GB')
"@

Write-Host ""
Write-Host "=== Done! ===" -ForegroundColor Green
Write-Host "Activate: .\$VenvDir\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "Then run: .\run_pipeline.ps1" -ForegroundColor White
