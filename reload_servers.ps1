Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  Sea Sentinel 2.0 - Reloading Modular Servers" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

# Terminate existing servers
Write-Host "[1/3] Terminating any existing servers on ports 8000 and 3000..." -ForegroundColor Yellow
@(8000, 3000) | ForEach-Object {
    $port = $_
    $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($connections) {
        $connections | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
            try {
                Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
                Write-Host "  Terminated process $_ on port $port" -ForegroundColor Green
            } catch {}
        }
    }
}

# Python executable resolution
$pythonExe = "python"
if (Test-Path "$PSScriptRoot\.venv\Scripts\python.exe") {
    $pythonExe = "$PSScriptRoot\.venv\Scripts\python.exe"
}

# Start Backend
Write-Host "[2/3] Starting FastAPI Backend on http://localhost:8000 ..." -ForegroundColor Yellow
Start-Process $pythonExe -ArgumentList "-m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload" -WorkingDirectory "$PSScriptRoot\backend"

# Start Frontend
Write-Host "[3/3] Starting Frontend Server on http://localhost:3000 ..." -ForegroundColor Yellow
Start-Process $pythonExe -ArgumentList "-m http.server 3000 --directory frontend" -WorkingDirectory "$PSScriptRoot"

Write-Host "`nServers successfully launched!" -ForegroundColor Green
Write-Host " - Web Dashboard:    http://localhost:3000" -ForegroundColor Cyan
Write-Host " - Backend API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host " - Architecture:     Production Modular Layout (frontend/ + backend/)" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
