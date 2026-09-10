@echo off
echo ===================================================
echo   Sea Sentinel - Reloading Backend and Frontend
echo ===================================================

echo [1/3] Stopping any processes listening on ports 8000 and 3000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a 2>nul
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a 2>nul
)

set PYTHON_EXE=python
if exist "%~dp0.venv\Scripts\python.exe" (
    set PYTHON_EXE=%~dp0.venv\Scripts\python.exe
)

echo [2/3] Starting FastAPI Backend Server on http://localhost:8000 ...
start "Sea Sentinel Backend" /D "%~dp0backend" "%PYTHON_EXE%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

echo [3/3] Starting Frontend Web Server on http://localhost:3000 ...
start "Sea Sentinel Frontend" /D "%~dp0" "%PYTHON_EXE%" -m http.server 3000 --directory frontend

echo.
echo Servers successfully reloaded!
echo - Web Dashboard:    http://localhost:3000
echo - Backend API Docs: http://localhost:8000/docs
echo ===================================================
