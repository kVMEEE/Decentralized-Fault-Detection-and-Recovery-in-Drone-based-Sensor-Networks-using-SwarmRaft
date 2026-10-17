import sys
import os
import math
import asyncio
from typing import Dict, List, Optional
import numpy as np

# Add parent directory and python_layer to path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "python_layer"))

from python_layer.fault_injector import ACTIVE_FAULT, FailureMode, FormationType
from python_layer.consensus import SwarmConsensusCluster, RaftRole
from python_layer.trajectory import PotentialFieldPlanner
import python_layer.swarmraft_native as native

# Try loading trained RL model
rl_model = None
try:
    from stable_baselines3 import PPO
    model_path = os.path.join(_ROOT, "python_layer", "models", "gdop_defense_policy.zip")
    if os.path.exists(model_path):
        rl_model = PPO.load(model_path)
except Exception:
    rl_model = None

# ==============================================================================
#  Simulation constants
# ==============================================================================
N_DRONES         = 8
FORMATION_RADIUS = 30.0
ASSET_POS        = np.array([50.0, 50.0, 15.0])
FORMATION_Z      = 26.0
Z_SPAN           = 7.0

HOLD_FRAMES      = 60
FAILURE_WINDOW   = 40
REORG_FRAMES     = 100
TAIL_FRAMES      = 40
TOTAL_FRAMES     = HOLD_FRAMES + FAILURE_WINDOW + REORG_FRAMES + TAIL_FRAMES

MECH_JITTER_LEN  = 30
CRASH_FALL_LEN   = 22

HONEST        = "honest"
FAULTING      = "faulting"
FAILED        = "failed"
REPOSITIONING = "repositioning"
BYZANTINE     = "byzantine"

# ==============================================================================
#  Formation slot functions (Full 3D Non-Coplanar Geometry)
# ==============================================================================
def ring_slots(n: int, radius: float, center: np.ndarray, z_base: float, z_span: float = Z_SPAN) -> np.ndarray:
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    z = z_base + z_span * np.sin(angles * 2)
    return np.stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
        z,
    ], axis=1)

def diamond_slots(n: int, radius: float, center: np.ndarray, z_base: float, z_span: float = Z_SPAN) -> np.ndarray:
    corners = np.array([
        [center[0],           center[1] + radius],
        [center[0] + radius,  center[1]           ],
        [center[0],           center[1] - radius],
        [center[0] - radius,  center[1]           ],
    ])
    t_vals = np.linspace(0, 1, n, endpoint=False)
    pts = []
    for i, t in enumerate(t_vals):
        edge = int(t * 4) % 4
        et   = (t * 4) - int(t * 4)
        p    = corners[edge] + et * (corners[(edge + 1) % 4] - corners[edge])
        z    = z_base + z_span * (1.0 if i % 2 == 0 else -1.0)
        pts.append([p[0], p[1], z])
    return np.array(pts)

def grid_slots(n: int, radius: float, center: np.ndarray, z_base: float, z_span: float = Z_SPAN) -> np.ndarray:
    cols = max(2, math.ceil(math.sqrt(n * 2)))
    rows = math.ceil(n / cols)
    dx = 2 * radius / (cols + 1)
    dy = 2 * radius / (rows + 1)
    ox = center[0] - radius
    oy = center[1] - radius
    pts = []
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if len(pts) >= n: break
            z = z_base + z_span * math.sin(idx * 1.3)
            pts.append([ox + (col + 1) * dx, oy + (row + 1) * dy, z])
            idx += 1
    return np.array(pts[:n])

def v_slots(n: int, radius: float, center: np.ndarray, z_base: float, z_span: float = Z_SPAN) -> np.ndarray:
    left_arm  = (n - 1) // 2
    right_arm = n - 1 - left_arm
    half_ang  = math.radians(55)
    front_y   = center[1] + radius * 0.7
    step      = radius / max(max(left_arm, right_arm), 1)
    pts = [[center[0], front_y, z_base + z_span]]
    for k in range(1, left_arm + 1):
        d = step * k
        z = z_base + z_span * math.cos(k * 0.8)
        pts.append([center[0] - d * math.sin(half_ang), front_y - d * math.cos(half_ang), z])
    for k in range(1, right_arm + 1):
        d = step * k
        z = z_base - z_span * math.cos(k * 0.8)
        pts.append([center[0] + d * math.sin(half_ang), front_y - d * math.cos(half_ang), z])
    return np.array(pts)

def compute_slots(ftype: FormationType, n: int, radius: float, center: np.ndarray, z_base: float, z_span: float = Z_SPAN) -> np.ndarray:
    fn = {
        FormationType.RING:    ring_slots,
        FormationType.DIAMOND: diamond_slots,
        FormationType.GRID:    grid_slots,
        FormationType.V_SHAPE: v_slots,
    }.get(ftype, ring_slots)
    return fn(n, radius, center, z_base, z_span)

def angular_assign(drone_pos: np.ndarray, slot_pos: np.ndarray, center: np.ndarray) -> List[int]:
    def angle_of(p):
        return math.atan2(p[1] - center[1], p[0] - center[0])
    n = len(drone_pos)
    d_order = sorted(range(n), key=lambda i: angle_of(drone_pos[i]))
    s_order = sorted(range(n), key=lambda i: angle_of(slot_pos[i]))
    result = [0] * n
    for rank, d_idx in enumerate(d_order):
        result[d_idx] = s_order[rank]
    return result

# ==============================================================================
#  Async Simulation Stream Generator
# ==============================================================================
async def run_simulation(fps=30):
    FAIL_START = HOLD_FRAMES
    FAIL_END   = HOLD_FRAMES + FAILURE_WINDOW
    fault = ACTIVE_FAULT.resolve(N_DRONES, FAIL_START, FAIL_END)
    FMT = fault.resolved_formation

    initial_slots = compute_slots(FMT, N_DRONES, FORMATION_RADIUS, ASSET_POS, FORMATION_Z, Z_SPAN)
    positions = initial_slots.copy()
    states = [HONEST] * N_DRONES

    consensus_cluster = SwarmConsensusCluster(N_DRONES)
    planner = PotentialFieldPlanner(d_safe=4.5, asset_radius=7.0)

    apf_trajectories = None
    new_slots_arr = None
    reorg_frame = None
    failure_done = False
    mech_counter = 0
    collision_vel = None
    events = []

    def log(msg: str):
        events.append(msg)
        if len(events) > 6:
            events.pop(0)

    for frame in range(TOTAL_FRAMES):
        # Tick Raft Consensus
        raft_logs = consensus_cluster.tick()
        for rl in raft_logs:
            log(rl)

        leader_id = consensus_cluster.leader_id
        fid = fault.failed_drone_id
        fmode = fault.failure_mode

        phase_name = ""

        # -- HOLD -------------------------------------------------------------
        if frame < HOLD_FRAMES:
            phase_name = f"ACTIVE DEFENSE [{FMT.value.upper()} 3D MESH]"
            if frame == 0:
                log(f"3D Swarm Online: {N_DRONES} drones in [{FMT.value}] configuration")
                log(f"Raft Consensus: Drone {leader_id} verified as LEADER (Term 1)")

        # -- FAILURE & BYZANTINE DETECTION ------------------------------------
        elif not failure_done:
            phase_name = "ADVERSARIAL FAULT DETECTED"
            if states[fid] != FAILED:
                if fmode == FailureMode.MECHANICAL:
                    if states[fid] != FAULTING:
                        states[fid] = FAULTING
                        log(f"Telemetry warning: Drone {fid} mechanical jitter")
                    mech_counter += 1
                    jitter = np.random.randn(3) * 1.5
                    jitter[2] = 0.0
                    positions[fid] = initial_slots[fid] + jitter
                    if mech_counter >= MECH_JITTER_LEN:
                        states[fid] = FAILED
                        consensus_cluster.kill_node(fid)
                        log(f"Heartbeat expired: Drone {fid} marked OFFLINE")
                elif fmode == FailureMode.RANDOM_COLLISION:
                    if collision_vel is None:
                        rng2 = np.random.default_rng()
                        collision_vel = np.array([float(rng2.uniform(-2.0, 2.0)),
                                                  float(rng2.uniform(-2.0, 2.0)),
                                                  -0.8])
                        states[fid] = FAULTING
                        log(f"Collision detected: Drone {fid} tumbling")
                    collision_vel[2] -= 0.3
                    positions[fid] += collision_vel + np.random.randn(3) * 0.2
                    if positions[fid][2] < 2.0 or frame >= fault.failure_frame + CRASH_FALL_LEN:
                        states[fid] = FAILED
                        consensus_cluster.kill_node(fid)
                        log(f"Drone {fid} crashed to ground")
                else:
                    if frame >= fault.failure_frame:
                        states[fid] = FAILED
                        consensus_cluster.kill_node(fid)
                        log(f"Byzantine attack detected: Drone {fid} eliminated")

            # Dynamic Byzantine cross-check
            if frame == fault.failure_frame - 2:
                pairwise = np.zeros((N_DRONES, N_DRONES))
                for i in range(N_DRONES):
                    for j in range(N_DRONES):
                        pairwise[i, j] = np.linalg.norm(positions[i] - positions[j])
                byz_nodes = native.detect_byzantine_nodes(positions, pairwise, threshold=3.0)
                if byz_nodes:
                    log(f"C++ BFT Quorum: Identified Byzantine nodes: {byz_nodes}")

            # Trigger Gap-Cover Reorganization
            if states[fid] == FAILED and not failure_done:
                failure_done = True
                survivors = [i for i in range(N_DRONES) if states[i] != FAILED]

                if leader_id == fid:
                    log(f"LEADER LOSS: Drone {fid} failed -> Raft election running")
                    for s in survivors:
                        if consensus_cluster.nodes[s].role == RaftRole.FOLLOWER:
                            consensus_cluster.nodes[s].timer = 0
                    consensus_cluster.tick()
                    leader_id = consensus_cluster.leader_id

                log(f"Leader Drone {leader_id} issuing Gap-Cover Reorganization")

                # Compute baseline slots
                new_slots_arr = compute_slots(FMT, len(survivors), FORMATION_RADIUS,
                                              ASSET_POS, FORMATION_Z, Z_SPAN)

                # RL formation optimization
                if rl_model is not None:
                    try:
                        obs = np.zeros(len(survivors) * 3 + 4, dtype=np.float32)
                        obs[:len(survivors)*3] = new_slots_arr.flatten()
                        obs[len(survivors)*3: len(survivors)*3 + 3] = ASSET_POS
                        obs[-1] = native.calculate_gdop(new_slots_arr, ASSET_POS)
                        action, _ = rl_model.predict(obs, deterministic=True)
                        offsets = action.reshape((len(survivors), 3))
                        new_slots_arr = new_slots_arr + np.clip(offsets, -1.5, 1.5)
                        log("RL Defense: PPO GDOP optimization applied")
                    except Exception:
                        pass

                surv_pos = np.array([positions[i] for i in survivors])
                assign   = angular_assign(surv_pos, new_slots_arr, ASSET_POS)

                starts_dict = {survivors[d]: positions[survivors[d]].copy() for d in range(len(survivors))}
                targets_dict = {survivors[d]: new_slots_arr[assign[d]] for d in range(len(survivors))}

                # APF collision-free trajectories
                apf_trajectories = planner.generate_trajectories(
                    starts_dict, targets_dict, ASSET_POS, n_frames=REORG_FRAMES
                )
                reorg_frame = frame
                for i in survivors:
                    states[i] = REPOSITIONING

        # -- REORGANIZATION ----------------------------------------------------
        else:
            if reorg_frame is not None and apf_trajectories is not None:
                elapsed = frame - reorg_frame
                f_idx = min(elapsed, REORG_FRAMES - 1)
                survivors = [i for i in range(N_DRONES) if states[i] != FAILED]
                for i in survivors:
                    if i in apf_trajectories:
                        positions[i] = apf_trajectories[i][f_idx]

                if elapsed >= REORG_FRAMES:
                    for i in survivors:
                        if states[i] == REPOSITIONING:
                            states[i] = HONEST
                    if frame == reorg_frame + REORG_FRAMES:
                        log("Reorganization complete -- 360-deg 3D defense restored")
                        final_gdop = native.calculate_gdop([positions[i] for i in survivors], ASSET_POS)
                        log(f"Optimized 3D GDOP: {final_gdop:.2f}")

            phase_name = f"GAP-COVER APF REORGANIZATION [{FMT.value.upper()}]"

        # Calculate metrics using C++ engine
        alive = [i for i in range(N_DRONES) if states[i] != FAILED]
        alive_ct = len(alive)
        coverage = int(round(alive_ct / N_DRONES * 100))
        gdop = native.calculate_gdop([positions[i] for i in alive], ASSET_POS)
        leader_id = consensus_cluster.leader_id

        state_dict = {
            "frame": frame,
            "total_frames": TOTAL_FRAMES,
            "phase": phase_name,
            "drones": [
                {
                    "id": i,
                    "state": states[i],
                    "pos": positions[i].tolist(),
                    "is_leader": (i == leader_id)
                } for i in range(N_DRONES)
            ],
            "asset_pos": ASSET_POS.tolist(),
            "target_slots": new_slots_arr.tolist() if new_slots_arr is not None else [],
            "metrics": {
                "gdop": round(gdop, 2),
                "coverage": coverage,
                "alive": alive_ct,
                "total": N_DRONES,
                "formation": FMT.value,
                "failure_mode": fault.failure_mode.value,
                "raft_term": consensus_cluster.nodes[max(0, leader_id)].current_term if leader_id >= 0 else 1
            },
            "logs": events.copy()
        }

        yield state_dict
        await asyncio.sleep(1 / fps)

    yield {"status": "complete"}
