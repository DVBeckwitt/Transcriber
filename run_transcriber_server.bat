@echo off
setlocal

if defined VIRTUAL_ENV (
  set "PY=%VIRTUAL_ENV%\Scripts\python.exe"
) else if exist "%USERPROFILE%\.venv-transcriber\Scripts\python.exe" (
  set "PY=%USERPROFILE%\.venv-transcriber\Scripts\python.exe"
) else (
  set "PY=%USERPROFILE%\.venv\Scripts\python.exe"
)

if not exist "%PY%" (
  echo Could not find Python at "%PY%"
  echo Fix: run .\install_transcriber_server.ps1, activate a venv, or set VIRTUAL_ENV.
  exit /b 2
)

if "%TRANSCRIBE_PROXY_TOKEN%"=="" (
  for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('TRANSCRIBE_PROXY_TOKEN', 'User')"`) do set "TRANSCRIBE_PROXY_TOKEN=%%T"
)

if "%TRANSCRIBE_PROXY_TOKEN%"=="" (
  echo TRANSCRIBE_PROXY_TOKEN is not set.
  echo Fix: set it in the user environment before starting the worker.
  echo   [Environment]::SetEnvironmentVariable("TRANSCRIBE_PROXY_TOKEN", "^<long-random-token^>", "User")
  exit /b 2
)

"%PY%" -c "import importlib.util, sys; missing=[m for m in ('whisperx','fastapi','uvicorn','multipart') if importlib.util.find_spec(m) is None]; print('Missing Python modules: ' + ', '.join(missing)) if missing else None; sys.exit(1 if missing else 0)"
if errorlevel 1 (
  echo Fix: install WhisperX and this package's server extras into the same virtual environment.
  echo   "%PY%" -m pip install -e ".[server]"
  exit /b 2
)

"%PY%" -m transcriber.server %*
