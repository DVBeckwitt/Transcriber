param(
    [string]$VenvPath = "$env:USERPROFILE\.venv-transcriber"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path $VenvPath)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $VenvPath
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $VenvPath
    } else {
        throw "Python was not found. Install Python 3.10+ first."
    }
}

$Python = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Could not find venv Python at $Python"
}

& $Python -m pip install --upgrade pip

Push-Location $RepoRoot
try {
    & $Python -m pip install -e ".[server]"
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Transcriber server extras installed in $VenvPath"
Write-Host "Install WhisperX and the right PyTorch build in the same venv before running real jobs."
Write-Host "Start the worker with:"
Write-Host "  .\run_transcriber_server.bat --host 0.0.0.0"
