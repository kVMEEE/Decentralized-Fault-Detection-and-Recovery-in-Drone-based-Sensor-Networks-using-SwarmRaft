@echo off
echo Cleaning up old processes...
taskkill /IM drone_agent2.exe /F 2>nul
taskkill /IM drone_agent.exe /F 2>nul
taskkill /FI "WINDOWTITLE eq Live SwarmRaft*" /F 2>nul
taskkill /FI "WINDOWTITLE eq Figure 1" /F 2>nul

echo Starting C++ Swarm Mesh Agents (Nodes 0 to 7)...
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

echo Launching Live Phase 1 3D Visualizer...
cd python_layer
c:\Users\sangh\Downloads\Swarm-Raft-main\.venv\Scripts\python.exe visuals_3d.py
