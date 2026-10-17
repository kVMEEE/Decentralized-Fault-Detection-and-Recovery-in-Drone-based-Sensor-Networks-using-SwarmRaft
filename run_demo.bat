@echo off
setlocal enabledelayedexpansion

echo =====================================================================
echo  SwarmRaft: Decentralized Fault Detection ^& Active Defense System
echo =====================================================================

REM Detect Python Environment with Fallbacks
set PYTHON_BIN=python
if exist ".venv\Scripts\python.exe" (
    set PYTHON_BIN=.venv\Scripts\python.exe
)

echo Using Python binary: %PYTHON_BIN%

echo Cleaning up old background processes...
taskkill /IM drone_agent2.exe /F 2>nul
taskkill /IM drone_agent.exe /F 2>nul
taskkill /FI "WINDOWTITLE eq Live SwarmRaft*" /F 2>nul
taskkill /FI "WINDOWTITLE eq Figure 1" /F 2>nul

set MODE=%1
if "%MODE%"=="" set MODE=sim

if /I "%MODE%"=="web" (
    echo Launching SwarmRaft Live Web Dashboard on http://localhost:8000
    %PYTHON_BIN% -m uvicorn web_app.server:app --host 0.0.0.0 --port 8000
    goto end
)

if /I "%MODE%"=="mesh" (
    echo Starting C++ Swarm Mesh Agents [Nodes 0 to 7]
    cd cpp_mesh
    start /b drone_agent2.exe 0 8
    start /b drone_agent2.exe 1 8
    start /b drone_agent2.exe 2 8
    start /b drone_agent2.exe 3 8
    start /b drone_agent2.exe 4 8
    start /b drone_agent2.exe 5 8
    start /b drone_agent2.exe 6 8
    start /b drone_agent2.exe 7 8
    cd ..
    echo Launching Live Telemetry 3D Visualizer
    cd python_layer
    %PYTHON_BIN% visuals_3d.py
    cd ..
    goto end
)

echo Starting Full Integrated SwarmRaft 3D Simulation
echo [Features: Real Raft Quorum, C++ BFT Quorum Filter, APF Collision-Free Reorganization, RL Defense]
%PYTHON_BIN% python_layer\swarm_sim.py

:end
endlocal
