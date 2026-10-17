# Decentralized Fault Detection and Recovery in Drone-Based Sensor Networks using SwarmRaft

## Project Overview

SwarmRaft is a decentralized UAV swarm coordination framework that maintains accurate positioning, fault tolerance, and operational continuity in GPS-denied or adversarial environments. The system simulates a fleet of 8 drones holding a 3D perimeter formation around a protected ground asset, surviving drone failures and actively reorganizing without any centralized controller.

Traditional drone swarms rely on GNSS and central command nodes, making them vulnerable to spoofing, jamming, and node failure. SwarmRaft addresses these challenges through:

- **Distributed Raft consensus** for decentralized leader election and term tracking
- **Geometric cross-ranging Byzantine fault detection** via C++/ctypes engine
- **3D multilateration** for GPS-denied position recovery
- **Artificial Potential Field (APF) trajectory planning** for certified collision-free reorganization
- **Reinforcement Learning (PPO)** for GDOP-optimized formation fine-tuning
- **Live 3D web dashboard** streaming simulation state over WebSocket

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        SwarmRaft Framework                       │
│                                                                  │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │  C++ Mesh Layer │  │   Python Layer   │  │   Web Layer    │  │
│  │─────────────────│  │──────────────────│  │────────────────│  │
│  │ swarmraft_core  │  │  consensus.py    │  │  server.py     │  │
│  │  - multilatera- │  │  (Raft quorum)   │  │  (FastAPI WS)  │  │
│  │    tion solver  │──│  trajectory.py   │──│  headless_sim  │  │
│  │  - Byzantine    │  │  (APF planner)   │  │  index.html    │  │
│  │    detection    │  │  swarm_sim.py    │  │  (Plotly 3D)   │  │
│  │  - GDOP calc   │  │  rl_defense_env  │  │                │  │
│  │  (ctypes bound) │  │  fault_injector  │  │                │  │
│  └─────────────────┘  └──────────────────┘  └────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### Decentralized Raft Consensus (`python_layer/consensus.py`)

- Full Raft leader election with randomized election timeouts (15–30 ticks)
- `RequestVote` RPCs with majority quorum enforcement across the alive cluster
- Periodic heartbeat (`AppendEntries`) broadcast from leader to reset follower timers
- On leader failure, followers time out and trigger a new election automatically
- Term tracking ensures stale leaders cannot win re-election

### Dynamic Byzantine Fault Detection (`python_layer/swarmraft_native.py`)

- Pairwise cross-ranging consistency check: each drone's reported position is compared against geometric distances computed from all peers' positions
- A node accumulates suspect votes if its position is inconsistent beyond a configurable threshold (default 3.0 m)
- Nodes with ≥ ⌊(n−1)/2⌋ votes are classified as Byzantine — **no drone IDs are hardcoded**
- Accelerated via C++ shared library (`cpp_mesh/swarmraft_core.dll`) through ctypes; pure-Python fallback activates automatically if DLL is unavailable

### 3D Multilateration Position Recovery (`python_layer/swarmraft_native.py`)

- Levenberg–Marquardt iterative solver reconstructs 3D position from ≥4 peer range measurements
- Handles GPS-spoofing and ranging attacks by excluding Byzantine-flagged anchors
- C++ engine (`solve_multilateration_3d`) provides sub-millisecond latency; Python fallback via NumPy available

### Collision-Free APF Trajectory Planning (`python_layer/trajectory.py`)

- Artificial Potential Field planner with:
  - Attractive force toward target formation slots
  - Repulsive force from protected asset (no-fly zone radius)
  - Inter-drone repulsive forces enforcing minimum separation (`d_safe = 4.5 m`)
  - Velocity-Verlet integration with sub-stepping for numerical stability
- `verify_safety()` certifies minimum separation across all frames post-generation

### RL Formation Optimizer (`python_layer/rl_defense_env.py`, `train_rl.py`)

- Custom `gymnasium` environment: action space = continuous displacement offsets per drone (3-DoF), observation = slot positions + asset + GDOP scalar
- Trained with `stable-baselines3` PPO for 15,000 timesteps using GDOP-minimization reward
- Saved policy (`python_layer/models/gdop_defense_policy.zip`) loaded at runtime to fine-tune slot assignments after each failure event

### Non-Coplanar 3D GDOP (`python_layer/swarmraft_native.py`, `web_app/headless_sim.py`)

- Formation slots carry altitude staggering (`z = z_base + Z_SPAN × sin(2θ)`), ensuring drones are never co-planar
- Non-coplanar geometry guarantees a well-conditioned `H^T H` matrix, producing valid GDOP values (verified: 1.5–3.5 for nominal 8-drone configuration)

### Fault Injection (`python_layer/fault_injector.py`)

Four configurable failure modes (edit `ACTIVE_FAULT` in `fault_injector.py`):

| Mode | Description |
|------|-------------|
| `RANDOM_ATTACK` | Instant destruction of a randomly chosen drone |
| `TARGETED_ATTACK` | Destroy a specific drone by ID |
| `RANDOM_COLLISION` | Physics-based spin-out crash over ~22 frames |
| `MECHANICAL` | Jitter telemetry for ~30 frames, then offline |

Four formation types: `RING`, `DIAMOND`, `GRID`, `V_SHAPE` (or `RANDOM` to pick each run).

---

## Quickstart

### Prerequisites

```bash
pip install -r requirements.txt
```

### Web Dashboard (recommended)

```bash
# Windows (batch)
.\run_demo.bat web

# Windows (PowerShell)
.\run_demo.ps1 web
```

Open **http://localhost:8000** — a live 3D Plotly visualization streams over WebSocket at 30 fps, showing drone states, Raft leader, GDOP, and phase transitions.

### Matplotlib Simulation

```bash
.\run_demo.bat
# or
.\run_demo.ps1
```

Runs the full integrated `python_layer/swarm_sim.py` with an animated 3D matplotlib window.

### C++ UDP Mesh (Windows only)

```bash
.\run_demo.bat mesh
```

Launches 8 `drone_agent2.exe` processes communicating over UDP on localhost with live telemetry fed into `python_layer/visuals_3d.py`.

### Docker (web dashboard only)

```bash
docker build -t swarmraft .
docker run -p 8000:8000 swarmraft
```

### Verification Test Suite

```bash
python test_integration.py
```

Runs 7 automated tests covering: 3D GDOP validity, Byzantine detection, C++ latency benchmark, Raft quorum election, APF safety verification, RL policy inference, and async WebSocket stream.

---

## Technologies

| Component | Technology |
|-----------|-----------|
| Consensus | Raft (Python, pure in-process simulation) |
| Byzantine Detection | Pairwise cross-ranging residuals (C++/ctypes + Python fallback) |
| Position Recovery | Levenberg–Marquardt multilateration (C++/ctypes + NumPy fallback) |
| GDOP | Geometric Dilution of Precision, 3D non-coplanar (C++/ctypes) |
| Trajectory Planning | Artificial Potential Fields, Velocity-Verlet integration (Python) |
| RL Optimization | PPO via stable-baselines3 + gymnasium |
| Visualization (2D/3D) | Matplotlib, mpl_toolkits.mplot3d |
| Web Dashboard | FastAPI, WebSocket, Plotly.js |
| Native Bridge | ctypes (Windows DLL; Python NumPy fallback on Linux/macOS) |

---

## Project Structure

```
SwarmRaft/
├── cpp_mesh/
│   ├── swarmraft_core.cpp       # C++ engine: multilateration, Byzantine detection, GDOP
│   ├── swarmraft_core.dll       # Pre-compiled Windows DLL (ctypes-bound)
│   ├── distributed_agent.cpp    # Standalone UDP mesh agent source
│   └── drone_agent2.exe         # Pre-compiled UDP mesh agent (Windows)
│
├── python_layer/
│   ├── fault_injector.py        # Scenario configuration (edit ACTIVE_FAULT here)
│   ├── consensus.py             # Full Raft consensus: election, quorum, heartbeat
│   ├── swarmraft_native.py      # ctypes bridge + Python fallbacks
│   ├── trajectory.py            # APF collision-free trajectory planner
│   ├── swarm_sim.py             # Integrated Matplotlib simulation
│   ├── rl_defense_env.py        # Gymnasium RL environment (GDOP reward)
│   ├── train_rl.py              # PPO training script
│   ├── visuals_3d.py            # Live UDP telemetry 3D visualizer
│   └── models/
│       └── gdop_defense_policy.zip  # Trained PPO policy
│
├── web_app/
│   ├── server.py                # FastAPI WebSocket server
│   ├── headless_sim.py          # Async simulation engine (no Matplotlib)
│   └── static/
│       └── index.html           # Live 3D Plotly dashboard
│
├── test_integration.py          # 7-test verification suite
├── requirements.txt             # Full dependencies
├── requirements-web.txt         # Minimal deps for Docker web deployment
├── Dockerfile                   # Web dashboard container
├── run_demo.bat                 # Windows launcher (batch)
└── run_demo.ps1                 # Windows launcher (PowerShell)
```

---

## Results

### Raft Leader Election
Drone 0 starts as leader (Term 1). On failure, followers time out and one wins a majority-quorum election (verified: new leader elected within a single tick in test suite).

### Byzantine Detection
Nodes reporting positions inconsistent with peer ranging measurements by > 3.0 m receive suspect votes from all peers; any node exceeding honest-majority threshold is flagged without any hardcoded ID.

### APF Trajectory Safety
7-drone crossing trajectory test (inverted targets) produces minimum separation > 3.0 m across all 80 frames, verified by `PotentialFieldPlanner.verify_safety()`.

### GDOP
Nominal 8-drone staggered ring produces GDOP ≈ 1.7–2.5 (well-conditioned). After 1-drone failure, 7-drone reconfigured formation GDOP is logged and displayed on the dashboard.

---

## Future Work

- Compile `swarmraft_core` as a Linux shared object (`.so`) for cross-platform C++ acceleration
- Multi-agent RL (MARL) for cooperative swarm optimization without a central RL step
- Real-world deployment on physical UAV platforms (e.g., PX4 / ArduPilot integration)
- Extend to full 3D altitude-aware ranging with UWB or acoustic sensors
