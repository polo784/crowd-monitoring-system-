@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Restore this environment with uv sync --frozen first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "%~dp0launch_dashboard.py" %*
if errorlevel 1 pause
