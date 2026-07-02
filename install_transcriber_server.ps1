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

& $Python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('whisperx') else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "WhisperX is not installed in this venv. Real transcription jobs will fail until WhisperX and a compatible PyTorch build are installed here."
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg was not found on PATH. The transcriber can run, but media preprocessing will fall back to the original source."
}

Write-Host "Use the same venv for WhisperX, PyTorch, and transcriber-local[server]."
Write-Host "Start the worker with:"
Write-Host "  .\run_transcriber_server.bat --host 0.0.0.0"
