param (
    [string]$Mode = "sim"
)

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  SwarmRaft: Decentralized Fault Detection & Active Defense System" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

# Detect Python Environment with Fallbacks
$PythonBin = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $PythonBin = ".venv\Scripts\python.exe"
}

Write-Host "Using Python binary: $PythonBin" -ForegroundColor Gray

# Clean up any lingering background processes
Write-Host "Cleaning up old background processes..." -ForegroundColor Gray
Get-Process drone_agent2, drone_agent -ErrorAction SilentlyContinue | Stop-Process -Force

if ($Mode -ieq "web") {
    Write-Host "`nLaunching SwarmRaft Live Web Dashboard on http://localhost:8000 ..." -ForegroundColor Green
    & $PythonBin -m uvicorn web_app.server:app --host 0.0.0.0 --port 8000
} elseif ($Mode -ieq "mesh") {
    Write-Host "`nStarting C++ Swarm Mesh Agents (Nodes 0 to 7)..." -ForegroundColor Yellow
    Push-Location cpp_mesh
    0..7 | ForEach-Object {
        Start-Process -FilePath ".\drone_agent2.exe" -ArgumentList "$_ 8" -NoNewWindow
    }
    Pop-Location
    Write-Host "Launching Live Telemetry 3D Visualizer..." -ForegroundColor Green
    Push-Location python_layer
    & $PythonBin visuals_3d.py
    Pop-Location
} else {
    Write-Host "`nStarting Full Integrated SwarmRaft 3D Simulation..." -ForegroundColor Green
    Write-Host "(Features: Real Raft Quorum, C++ BFT Quorum Filter, APF Collision-Free Reorganization, RL Defense)" -ForegroundColor Gray
    & $PythonBin python_layer\swarm_sim.py
}
