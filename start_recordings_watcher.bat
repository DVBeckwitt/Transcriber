@echo off
setlocal EnableExtensions EnableDelayedExpansion

pushd "%~dp0"

set "WATCH_DIR=%USERPROFILE%\OneDrive\recordings"
set "PID_FILE=%WATCH_DIR%\transcriber-watcher.pid"
set "VENV=%USERPROFILE%\.venv"
if defined VIRTUAL_ENV set "VENV=%VIRTUAL_ENV%"
set "PY=%VENV%\Scripts\python.exe"

if not exist "%PY%" (
  echo.
  echo Could not find Python at:
  echo   "%PY%"
  echo Fix: set VIRTUAL_ENV, or install whisperx into %USERPROFILE%\.venv
  echo.
  pause
  popd
  exit /b 1
)

if not exist "%WATCH_DIR%" mkdir "%WATCH_DIR%" >nul 2>&1
if not exist "%WATCH_DIR%" (
  echo.
  echo Could not create or access watch directory:
  echo   "%WATCH_DIR%"
  echo.
  pause
  popd
  exit /b 1
)

if exist "%PID_FILE%" (
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$lines = Get-Content '%PID_FILE%' -ErrorAction SilentlyContinue; if($null -eq $lines -or $lines.Count -eq 0){ exit 1 }; $watcherId = [int]$lines[0].Trim(); $expectedStart = $null; if($lines.Count -gt 1 -and -not [string]::IsNullOrWhiteSpace($lines[1])){ $expectedStart = [int64]$lines[1].Trim() }; $p = Get-Process -Id $watcherId -ErrorAction SilentlyContinue; if($null -eq $p -or $p.ProcessName -ne 'python'){ exit 1 }; if($expectedStart -ne $null -and $p.StartTime.ToUniversalTime().ToFileTimeUtc() -ne $expectedStart){ exit 1 }; exit 0"
  if not errorlevel 1 (
    set "WATCHER_PID="
    set /p WATCHER_PID=<"%PID_FILE%"
    echo.
    echo Watcher is already running with PID !WATCHER_PID!.
    echo Watcher log:
    echo   "%WATCH_DIR%\transcriber-watcher.log"
    echo To stop it:
    echo   ".\stop_recordings_watcher.bat"
    echo.
    popd
    endlocal & exit /b 0
  )
  del "%PID_FILE%" >nul 2>&1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p = Start-Process -WindowStyle Hidden -WorkingDirectory '%CD%' -FilePath '%PY%' -ArgumentList @('-m','transcriber','--watch','--watch-dir','%WATCH_DIR%','--lang','en','--task','transcribe','--mode','quality') -PassThru; [System.IO.File]::WriteAllLines('%PID_FILE%', @([string]$p.Id, [string]$p.StartTime.ToUniversalTime().ToFileTimeUtc()), [System.Text.Encoding]::ASCII)"
set "RC=%ERRORLEVEL%"

if "%RC%"=="0" (
  set "WATCHER_PID="
  set /p WATCHER_PID=<"%PID_FILE%"
  echo.
  echo Watcher started for:
  echo   "%WATCH_DIR%"
  echo Watcher PID:
  echo   "!WATCHER_PID!"
  echo Watcher log:
  echo   "%WATCH_DIR%\transcriber-watcher.log"
  echo To stop it:
  echo   ".\stop_recordings_watcher.bat"
  echo.
)

popd
endlocal & exit /b %RC%
