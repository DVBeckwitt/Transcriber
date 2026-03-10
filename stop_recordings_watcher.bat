@echo off
setlocal EnableExtensions EnableDelayedExpansion

pushd "%~dp0"

set "WATCH_DIR=%USERPROFILE%\OneDrive\recordings"
set "PID_FILE=%WATCH_DIR%\transcriber-watcher.pid"
set "WATCH_MATCH=* -m transcriber*--watch*--watch-dir*%WATCH_DIR%*"
set "RC=1"

if exist "%PID_FILE%" (
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$lines = Get-Content '%PID_FILE%' -ErrorAction SilentlyContinue; if($null -eq $lines -or $lines.Count -eq 0){ exit 2 }; $watcherId = [int]$lines[0].Trim(); $expectedStart = $null; if($lines.Count -gt 1 -and -not [string]::IsNullOrWhiteSpace($lines[1])){ $expectedStart = [int64]$lines[1].Trim() }; $p = Get-Process -Id $watcherId -ErrorAction SilentlyContinue; if($null -eq $p -or $p.ProcessName -ne 'python'){ exit 2 }; if($expectedStart -ne $null -and $p.StartTime.ToUniversalTime().ToFileTimeUtc() -ne $expectedStart){ exit 2 }; Stop-Process -Id $watcherId -Force; exit 0"
  if not errorlevel 1 (
    del "%PID_FILE%" >nul 2>&1
    echo.
    echo Watcher stopped.
    echo.
    popd
    endlocal & exit /b 0
  )
  del "%PID_FILE%" >nul 2>&1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { $procs = @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '%WATCH_MATCH%' }); if($procs.Count -eq 0){ exit 3 }; $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop }; exit 0 } catch { exit 4 }"
set "RC=%ERRORLEVEL%"

if "%RC%"=="0" (
  del "%PID_FILE%" >nul 2>&1
  echo.
  echo Watcher stopped.
  echo.
) else if "%RC%"=="3" (
  echo.
  echo No hidden watcher is currently running for:
  echo   "%WATCH_DIR%"
  echo.
) else (
  echo.
  echo Could not stop the hidden watcher automatically.
  echo Try this PowerShell command:
  echo   Get-CimInstance Win32_Process ^| Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '%WATCH_MATCH%' } ^| ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
  echo.
)

popd
endlocal & exit /b %RC%
