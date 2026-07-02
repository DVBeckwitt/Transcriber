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
  echo TRANSCRIBE_PROXY_TOKEN is not set. The server will fail closed unless --proxy-token is passed.
)

"%PY%" -m transcriber.server %*
