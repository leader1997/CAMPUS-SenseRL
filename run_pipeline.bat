@echo off
REM One-click CAMPUS-SenseRL full pipeline (Windows)
cd /d "%~dp0\.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\run_full_pipeline.py %*
) else (
  python scripts\run_full_pipeline.py %*
)
pause
