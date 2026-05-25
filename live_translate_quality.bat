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

if not defined LIVE_MODEL set "LIVE_MODEL=medium"
if not defined LIVE_NLLB_SIZE set "LIVE_NLLB_SIZE=600M"
if not defined LIVE_CHUNK_MS set "LIVE_CHUNK_MS=750"
if not defined LIVE_FRAME_THRESHOLD set "LIVE_FRAME_THRESHOLD=45"
if not defined LIVE_BEAMS set "LIVE_BEAMS=5"
if not defined LIVE_AUDIO_MIN_LEN set "LIVE_AUDIO_MIN_LEN=1.0"
if not defined LIVE_AUDIO_MAX_LEN set "LIVE_AUDIO_MAX_LEN=45"

set "LIVE_GLOSSARY_ARGS="
if exist "transcriber_glossary.txt" (
  set LIVE_GLOSSARY_ARGS=--glossary-file "transcriber_glossary.txt"
)

"%PY%" -m transcriber ^
  --live ^
  --lang es ^
  --model "%LIVE_MODEL%" ^
  --live-source system ^
  --no-speaker-labels ^
  --live-preset quality ^
  --live-translation-mode cascade ^
  --live-backend faster-whisper ^
  --live-chunk-ms "%LIVE_CHUNK_MS%" ^
  --live-frame-threshold "%LIVE_FRAME_THRESHOLD%" ^
  --live-beams "%LIVE_BEAMS%" ^
  --live-decoder beam ^
  --live-audio-min-len "%LIVE_AUDIO_MIN_LEN%" ^
  --live-audio-max-len "%LIVE_AUDIO_MAX_LEN%" ^
  --live-nllb-backend ctranslate2 ^
  --live-nllb-size "%LIVE_NLLB_SIZE%" ^
  --live-save-transcript "logs\live_english_transcript.txt" ^
  --live-save-bilingual-transcript "logs\live_bilingual_transcript.txt" ^
  --live-audio-diagnostics ^
  %LIVE_GLOSSARY_ARGS%
set "RC=%ERRORLEVEL%"

echo.
pause
popd
endlocal & exit /b %RC%
