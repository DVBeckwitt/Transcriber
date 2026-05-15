@echo off
setlocal EnableExtensions

pushd "%~dp0"

if defined VIRTUAL_ENV (
  set "PY=%VIRTUAL_ENV%\Scripts\python.exe"
) else if exist "%~dp0.uv-venv\Scripts\python.exe" (
  set "PY=%~dp0.uv-venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
  set "PY=%~dp0.venv\Scripts\python.exe"
) else (
  set "PY=%USERPROFILE%\.venv\Scripts\python.exe"
)

if not exist "%PY%" (
  echo.
  echo Could not find Python at:
  echo   "%PY%"
  echo Fix: set VIRTUAL_ENV, run uv sync --extra live, or install live dependencies with:
  echo   pip install -e ".[live]"
  echo.
  pause
  popd
  exit /b 1
)

"%PY%" -m transcriber --live --lang es --model small --live-source system --no-speaker-labels --live-save-bilingual-transcript "logs\live_bilingual_transcript.txt"
set "RC=%ERRORLEVEL%"

echo.
pause
popd
endlocal & exit /b %RC%
