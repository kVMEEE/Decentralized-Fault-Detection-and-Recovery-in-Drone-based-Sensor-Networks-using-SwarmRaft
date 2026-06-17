# DECENTRALIZED FAULT DETECTION AND RECOVERY IN DRONE-BASED SENSOR NETWORKS USING SWARMRAFT

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

**The system is designed to handle up to **$f$** Byzantine faults in a swarm of size **$n$**, provided that $n \ge 2f + 1$**. **By offloading the math-heavy verification to C, the leader node can process 17+ drones with sub-millisecond latency**.


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
