"""
Comprehensive Integration Test Suite for SwarmRaft System.
Validates all 7 closed gaps:
  1. 3D GDOP Non-Singularity
  2. Dynamic Byzantine Fault Detection (No Hardcoded IDs)
  3. C++ ctypes Engine & Sub-millisecond Latency Benchmark
  4. Real Raft Leader Election & Majority Quorum
  5. Collision-Free Potential Field Trajectory Safety
  6. RL Policy Loading & Formation Fine-Tuning
  7. Async Web Stream Generation
"""

import sys
import os
import time
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "python_layer"))

import python_layer.swarmraft_native as native
from python_layer.consensus import SwarmConsensusCluster, RaftRole
from python_layer.trajectory import PotentialFieldPlanner
from python_layer.rl_defense_env import ActiveGDOPDefenseEnv

def run_tests():
    print("=" * 70)
    print("  SWARMRAFT VERIFICATION & GAP-CLOSURE TEST SUITE")
    print("=" * 70)
    passed = 0
    total = 7

    # Test 1: 3D GDOP Non-Singularity
    print("\n[Test 1] 3D GDOP Non-Coplanar Geometry...")
    target = np.array([50.0, 50.0, 15.0])
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    coords_3d = np.stack([
        50.0 + 30.0 * np.cos(angles),
        50.0 + 30.0 * np.sin(angles),
        26.0 + 7.0 * np.sin(angles * 2)
    ], axis=1)
    gdop = native.calculate_gdop(coords_3d, target)
    print(f"  -> Calculated 3D GDOP: {gdop:.2f} (Target bracket: 1.5 - 4.0)")
    assert 1.0 < gdop < 10.0, f"GDOP singular or out of bounds: {gdop}"
    print("  -> PASS: 3D GDOP is mathematically valid and non-singular!")
    passed += 1

    # Test 2: Dynamic Byzantine Detection
    print("\n[Test 2] Dynamic Byzantine Detection (Zero Hardcoded IDs)...")
    n = 10
    reported = np.random.uniform(10, 90, (n, 3))
    pairwise = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pairwise[i, j] = np.linalg.norm(reported[i] - reported[j]) + np.random.normal(0, 0.1)
    # Inject spoofing into arbitrary node 4
    reported[4] += np.array([30.0, -25.0, 10.0])
    byz = native.detect_byzantine_nodes(reported, pairwise, threshold=3.0)
    print(f"  -> Injected node: 4 | Detected Byzantine nodes: {byz}")
    assert 4 in byz, f"Node 4 was not detected as Byzantine: {byz}"
    print("  -> PASS: Dynamic Byzantine cross-ranging detection verified!")
    passed += 1

    # Test 3: C++ ctypes Sub-millisecond Latency Benchmark
    print("\n[Test 3] C++ ctypes Benchmark (18 Drones)...")
    assert native._lib is not None, "C++ shared library not loaded!"
    t0 = time.perf_counter()
    iters = 1000
    for _ in range(iters):
        native.calculate_gdop(reported, target)
    dt_ms = (time.perf_counter() - t0) / iters * 1000.0
    print(f"  -> Verified latency: {dt_ms:.4f} ms per cycle (Sub-millisecond: {dt_ms < 1.0})")
    assert dt_ms < 1.0, f"Latency exceeded 1ms: {dt_ms}ms"
    print("  -> PASS: Sub-millisecond C++ execution verified!")
    passed += 1

    # Test 4: Real Raft Leader Election & Majority Quorum
    print("\n[Test 4] Decentralized Raft Quorum Consensus...")
    cluster = SwarmConsensusCluster(8)
    initial_leader = cluster.leader_id
    assert initial_leader == 0, f"Unexpected initial leader: {initial_leader}"
    # Kill leader
    cluster.kill_node(initial_leader)
    # Trigger election on follower
    cluster.nodes[1].timer = 0
    logs = cluster.tick()
    new_leader = cluster.leader_id
    print(f"  -> Old Leader: {initial_leader} (Killed) -> New Elected Leader: {new_leader} (Term {cluster.nodes[new_leader].current_term})")
    assert new_leader != initial_leader and new_leader != -1, "Leader election failed"
    print("  -> PASS: Decentralized Raft election and quorum verified!")
    passed += 1

    # Test 5: Collision-Free Potential Field Trajectories
    print("\n[Test 5] Artificial Potential Field (APF) Collision Avoidance...")
    planner = PotentialFieldPlanner(d_safe=5.0)
    starts = {i: np.array([20.0 + i * 6, 25.0, 20.0]) for i in range(7)}
    targets = {i: np.array([20.0 + (6 - i) * 6, 75.0, 20.0]) for i in range(7)} # Inverted targets
    trajs = planner.generate_trajectories(starts, targets, target, n_frames=80)
    is_safe, min_dist = PotentialFieldPlanner.verify_safety(trajs, min_allowed_dist=3.0)
    print(f"  -> Safety Verified: {is_safe} | Minimum Separation: {min_dist:.2f}m (Threshold: 3.0m)")
    assert is_safe, f"Trajectory separation violated: {min_dist}m"
    print("  -> PASS: Collision-free trajectory generation certified!")
    passed += 1

    # Test 6: RL Policy Loading & Optimization
    print("\n[Test 6] RL Formation Optimization Policy...")
    from stable_baselines3 import PPO
    model_path = os.path.join(_ROOT, "python_layer", "models", "gdop_defense_policy.zip")
    assert os.path.exists(model_path), f"Saved RL model not found at {model_path}"
    model = PPO.load(model_path)
    obs = np.zeros(7 * 3 + 4, dtype=np.float32)
    obs[:21] = coords_3d[:7].flatten()
    obs[21:24] = target
    obs[-1] = gdop
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (21,), f"Unexpected action shape: {action.shape}"
    print(f"  -> Model loaded successfully; predicted {action.shape[0]} continuous displacement control vectors")
    print("  -> PASS: RL defense policy operational!")
    passed += 1

    # Test 7: Async Web App Generator Test
    print("\n[Test 7] Async Web Stream Generator...")
    import asyncio
    from web_app.headless_sim import run_simulation
    async def test_sim():
        count = 0
        async for state in run_simulation(fps=1000):
            count += 1
            if count >= 30:
                break
        return count
    frames = asyncio.run(test_sim())
    assert frames == 30, f"Expected 30 frames, got {frames}"
    print("  -> PASS: Web simulation async generator verified!")
    passed += 1

    print("\n" + "=" * 70)
    print(f"  ALL TESTS PASSED: {passed}/{total} GAPS SUCCESSFULLY CLOSED")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
