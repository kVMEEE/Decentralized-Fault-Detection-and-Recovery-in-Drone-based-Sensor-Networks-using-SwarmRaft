
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

## 🚁 Project Overview

SwarmRaft is a decentralized UAV swarm coordination framework designed to maintain accurate positioning, fault tolerance, and operational continuity in GPS-denied or adversarial environments.

Traditional drone swarms rely heavily on GNSS signals and centralized controllers, making them vulnerable to spoofing attacks, signal jamming, and node failures. SwarmRaft addresses these challenges through a combination of distributed consensus, geometric position recovery, and AI-driven formation optimization.

The framework enables drones to collaboratively identify malicious nodes, recover compromised positions using peer-to-peer ranging information, and autonomously reorganize swarm formations after drone failures. Additionally, a Reinforcement Learning (RL) agent continuously optimizes swarm geometry to improve localization accuracy and maintain robust coverage.

---

## ✨ Key Features

### 🗳️ Decentralized Consensus & Fault Detection

* Implements a Raft-inspired consensus mechanism for leader election and distributed decision-making.
* Detects malicious or spoofed drones through collective voting and consistency verification.
* Maintains fault tolerance using a Byzantine-inspired voting strategy requiring honest-majority validation.

### 📍 Position Recovery & Localization

* Recovers corrupted drone positions using iterative multilateration based on neighboring drone distances.
* Supports dead-reckoning fallback when insufficient neighboring drones are available.
* Accurately reconstructs positions even under GPS spoofing and ranging attacks.

### 🤖 AI-Driven Formation Optimization

* Trains a Reinforcement Learning agent to optimize swarm formations.
* Uses Geometric Dilution of Precision (GDOP) as the primary optimization metric.
* Learns drone placement strategies that maximize localization accuracy while minimizing repositioning costs.
* Trained over 15,000 simulation episodes using reward-based optimization.

### 🔄 Autonomous Failure Recovery

* Detects drone crashes, communication loss, and node dropouts in real time.
* Executes Gap-Cover Reorganization to restore formation integrity.
* Computes collision-free trajectories for surviving drones and maintains continuous coverage.

### 🛡️ Adversarial Resilience

* Simulates GPS spoofing attacks, targeted node compromise, random attacks, and mechanical failures.
* Evaluates system robustness under hostile and uncertain operating conditions.
* Demonstrates reliable operation despite compromised drones and network disruptions.

### 📊 Real-Time Visualization & Analytics

* Provides live 3D swarm telemetry and recovery visualization.
* Displays true, spoofed, and recovered drone positions for performance evaluation.
* Supports large-scale simulation experiments and scalability analysis.

---

## 🏗️ System Architecture

The SwarmRaft framework consists of three interconnected layers:

### Communication Network Layer

* Handles drone-to-drone communication.
* Executes consensus voting and leader election.
* Manages fault detection and position recovery workflows.

### AI Defense Layer

* Hosts the Reinforcement Learning environment.
* Optimizes drone formations based on GDOP feedback.
* Continuously improves swarm localization performance.

### Visualization Layer

* Provides real-time monitoring of swarm behavior.
* Visualizes attacks, failures, recovery procedures, and AI training progress.

---

## 📈 Results

### GPS Spoofing Recovery

Successfully identifies drones reporting false GNSS coordinates and reconstructs their true positions through multilateration and consensus-driven verification.

### Autonomous Failure Handling

Detects drone failures in real time and performs dynamic Gap-Cover Reorganization, enabling the swarm to maintain coverage and formation stability without human intervention.

### AI-Based Formation Optimization

The Reinforcement Learning agent learns optimal drone arrangements that consistently reduce GDOP and improve localization precision across varying swarm configurations.

---

## 🚀 Technologies Used

* Python
* C
* Reinforcement Learning
* Distributed Systems
* Raft Consensus Algorithm
* Multilateration
* Sensor Fusion
* NumPy
* Matplotlib
* ctypes
* Monte Carlo Simulation

---

## 🔮 Future Work

* Extend localization to full 3D altitude-aware swarm coordination.
* Integrate higher-precision ranging sensors for improved recovery accuracy.
* Develop fully decentralized multi-agent reinforcement learning (MARL) for cooperative swarm optimization.
* Enable real-world deployment on physical UAV platforms.
