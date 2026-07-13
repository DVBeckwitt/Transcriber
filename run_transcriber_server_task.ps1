param(
    [string]$PythonPath = "$env:USERPROFILE\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDirectory = Join-Path $RepoRoot "logs"
$LogPath = Join-Path $LogDirectory "transcriber-server.log"
$StdoutLogPath = Join-Path $LogDirectory "transcriber-server.stdout.log"
$StderrLogPath = Join-Path $LogDirectory "transcriber-server.stderr.log"

# Task Scheduler can retain an older environment block. Refresh only the
# user-scoped values the worker consumes, without putting secrets in the task.
$EnvironmentNames = @(
    "HF_TOKEN",
    "TRANSCRIBE_COMPUTE_TYPE",
    "TRANSCRIBE_DEVICE",
    "TRANSCRIBE_FFMPEG",
    "TRANSCRIBE_JOB_TTL_SECONDS",
    "TRANSCRIBE_MAX_PENDING_JOBS",
    "TRANSCRIBE_MAX_UPLOAD_BYTES",
    "TRANSCRIBE_MAX_WORKERS",
    "TRANSCRIBE_PROXY_TOKEN",
    "TRANSCRIBE_SERVER_HOST",
    "TRANSCRIBE_SERVER_PORT",
    "TRANSCRIBE_WARM_VRAM",
    "TRANSCRIBE_WORK_DIR"
)

foreach ($Name in $EnvironmentNames) {
    $Value = [Environment]::GetEnvironmentVariable($Name, "User")
    if (-not [string]::IsNullOrWhiteSpace($Value)) {
        [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
    }
}

New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null

try {
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        throw "Could not find Python at $PythonPath"
    }
    if ([string]::IsNullOrWhiteSpace($env:TRANSCRIBE_PROXY_TOKEN)) {
        throw "TRANSCRIBE_PROXY_TOKEN is not set in the user environment."
    }

    Set-Location $RepoRoot
    $env:PYTHONUNBUFFERED = "1"
    Add-Content -LiteralPath $LogPath -Value "[$(Get-Date -Format o)] Starting Transcriber worker."
    $Process = Start-Process `
        -FilePath $PythonPath `
        -ArgumentList @("-u", "-m", "transcriber.server") `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $StdoutLogPath `
        -RedirectStandardError $StderrLogPath `
        -NoNewWindow `
        -Wait `
        -PassThru
    $ExitCode = $Process.ExitCode
    Add-Content -LiteralPath $LogPath -Value "[$(Get-Date -Format o)] Transcriber worker exited with code $ExitCode."
    exit $ExitCode
} catch {
    Add-Content -LiteralPath $LogPath -Value "[$(Get-Date -Format o)] Startup failed: $($_.Exception.Message)"
    exit 1
}
