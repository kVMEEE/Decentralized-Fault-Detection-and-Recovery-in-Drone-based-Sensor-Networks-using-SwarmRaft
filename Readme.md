
# SwarmRaft: Robust Drone Swarm Coordination

## 🚁 Project Overview

**SwarmRaft** is a decentralized coordination and positioning framework designed for Unmanned Aerial Vehicle (UAV) swarms operating in environments where Global Navigation Satellite Systems (GNSS) are degraded, spoofed, or denied.

**Standard drones rely heavily on GPS, which is vulnerable to jamming and environmental interference**. This project implements a **blockchain-inspired consensus mechanism** that allows a swarm to detect "liars" (spoofed drones) and recover their true positions using peer-to-peer distance measurements and the Raft consensus algorithm.

---

## 🛠️ Key Functionalities

* **Sensor Fusion Simulation** **: Every drone manages three "realities": Ground Truth (actual), GNSS (noisy/spoofed), and INS (inertial dead-reckoning with drift)**.
* **Adversarial Testing** **: Includes a dedicated module to inject GNSS spoofing and ranging-tampering attacks on a subset (**$f$**) of the swarm**.
* **Stage 1: Consistency Voting (C-Core)** **: A high-performance logic that compares reported coordinates against physical inter-node distances to flag faulty drones**.
* **Stage 2: Position Recovery (C-Core)** **: Uses iterative multilateration to re-calculate the position of flagged drones based only on data from "honest" neighbors**.
* **Scaling Analysis** **: Automated Monte Carlo simulations to prove that as the swarm grows, the recovery error drops significantly**.

---

## 📂 Folder Structure

**Plaintext**

```
SwarmRaft_Simulation/
├── core/                   # High-performance C Logic
│   ├── build/              # Compiled shared library (libswarmraft.so)
│   ├── swarmraft.c         # C Implementation of Voting & Recovery logic
│   └── swarmraft.h         # C Header defining the Python-to-C interface
├── simulation/             # Python Simulation Environment
│   ├── drone.py            # Individual drone sensor & state models
│   ├── world.py            # Physics engine & swarm movement coordination
│   ├── attacks.py          # Logic for injecting spoofing & interference
│   └── bridge.py           # The ctypes bridge connecting Python to the C Core
├── results/                # Directory for generated plots and MAE logs
├── main.py                 # Single-step visual test script
├── analysis.py             # Statistical scaling & performance analysis script
└── requirements.txt        # Python library dependencies
```

---

## 🚀 How to Run

### 1. Prerequisites

Install the required dependencies.

**Note:** *Make sure that the virtual environment is active. I used Anaconda to create a venv for me.*

`pip install -r requirements.txt`

To create a virtual environment using conda:

`conda create -p venv python==3.11 -y`

`conda activate venv/`

Then run the install command.

### 2. Compilation (The C Core)

Before running the simulation, you must compile the C logic into a shared library so Python can call it. Run this in your terminal:

**PowerShell**

```
gcc -shared -o core/build/libswarmraft.so -fPIC core/swarmraft.c
```

### 3. Run a Visual Test

To see a single time-step where drones are spoofed and then recovered (indicated by yellow stars):

**PowerShell**

```
python main.py
```

### 4. Run Scaling Analysis

To generate the logarithmic error scaling graph (Figure 2 in the paper) and see how accuracy improves with swarm size:

**PowerShell**

```
python analysis.py
```

---

## 📊 Performance Metrics

**The system is designed to handle up to **$f$** Byzantine faults in a swarm of size **$n$**, provided that $n \ge 2f + 1$**. **By offloading the math-heavy verification to C, the leader node can process 17+ drones with sub-millisecond latency**.
