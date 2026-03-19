@echo off
setlocal EnableExtensions

pushd "%~dp0"

set "WATCH_DIR=%USERPROFILE%\OneDrive\recordings"
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

"%PY%" -m transcriber --watch --watch-dir "%WATCH_DIR%" --lang auto --mode quality %*
set "RC=%ERRORLEVEL%"

echo.
pause
popd
endlocal & exit /b %RC%
