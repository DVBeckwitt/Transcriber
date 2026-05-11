@echo off
setlocal EnableExtensions

pushd "%~dp0"

set "WATCH_DIR=%USERPROFILE%\OneDrive\recordings"
set "EXTRA_WATCH_DIR=%USERPROFILE%\Videos\escuela"
set "ESCUELA_DEST=\\BECKWITT-SERVER\Plex\TV\Escuela de Nada"
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

echo.
echo Watching:
echo   "%WATCH_DIR%"
echo   "%EXTRA_WATCH_DIR%" ^(video only, Spanish to English SRT, no diarization names, moved to "%ESCUELA_DEST%"^)
echo.

"%PY%" -m transcriber --watch --watch-dir "%WATCH_DIR%" --lang auto --mode quality %*
set "RC=%ERRORLEVEL%"

echo.
pause
popd
endlocal & exit /b %RC%
