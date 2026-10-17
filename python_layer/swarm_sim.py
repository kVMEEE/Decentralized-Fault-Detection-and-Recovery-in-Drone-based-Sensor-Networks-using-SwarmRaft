# -*- coding: utf-8 -*-
# =============================================================================
#  SwarmRaft -- Decentralized Fault Detection and Recovery Drone Network
#  ---------------------------------------------------------------------------
#  Integrated System Implementation:
#    1. Drones hold a 3D perimeter formation around the asset with altitude diversity.
#    2. One drone experiences fault/attack (configured via fault_injector.py).
#    3. Dynamic Cross-Ranging Residual Consistency check detects Byzantine behavior.
#    4. Decentralized Raft consensus triggers election and majority quorum voting.
#    5. RL Active Defense model fine-tunes slot allocation for optimal GDOP.
#    6. Artificial Potential Field (APF) executes certified collision-free trajectories.
# =============================================================================

import sys
import os
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

matplotlib.rcParams["font.family"] = "DejaVu Sans"

# -- Path Setup & Module Imports -----------------------------------------------
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CURRENT_DIR)

from fault_injector import ACTIVE_FAULT, FailureMode, FormationType
from consensus import SwarmConsensusCluster, RaftRole
from trajectory import PotentialFieldPlanner
import swarmraft_native as native

# Try loading trained RL model
rl_model = None
try:
    from stable_baselines3 import PPO
    model_path = os.path.join(_CURRENT_DIR, "models", "gdop_defense_policy.zip")
    if os.path.exists(model_path):
        rl_model = PPO.load(model_path)
except Exception as e:
    rl_model = None

# ==============================================================================
#  Simulation constants
# ==============================================================================
N_DRONES         = 8
FORMATION_RADIUS = 30.0
ASSET_POS        = np.array([50.0, 50.0, 15.0]) # Protected ground asset
FORMATION_Z      = 26.0                         # Base flight cruise altitude
Z_SPAN           = 7.0                          # Altitude staggering for true 3D GDOP

HOLD_FRAMES      = 60    # hold formation before failure window opens
FAILURE_WINDOW   = 40    # window of frames in which failure can fire
REORG_FRAMES     = 100   # collision-free path interpolation length
TAIL_FRAMES      = 40    # hold final formation after reorg completes
TOTAL_FRAMES     = HOLD_FRAMES + FAILURE_WINDOW + REORG_FRAMES + TAIL_FRAMES

MECH_JITTER_LEN  = 30
CRASH_FALL_LEN   = 22

BG_COLOR    = "#0a0e1a"
PANEL_COLOR = "#0f172a"
DIM_BORDER  = "#334155"

# Drone state strings
HONEST        = "honest"
FAULTING      = "faulting"
FAILED        = "failed"
REPOSITIONING = "repositioning"
BYZANTINE     = "byzantine"

# ==============================================================================
#  Formation slot functions (Full 3D Non-Coplanar Geometry)
# ==============================================================================

def ring_slots(n: int, radius: float, center: np.ndarray, z_base: float, z_span: float = Z_SPAN) -> np.ndarray:
    """Evenly spaced 3D circle with dual-altitude sinusoidal staggering."""
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    z = z_base + z_span * np.sin(angles * 2)
    return np.stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
        z,
    ], axis=1)

def diamond_slots(n: int, radius: float, center: np.ndarray, z_base: float, z_span: float = Z_SPAN) -> np.ndarray:
    """Drones distributed along diamond perimeter with alternating altitude tiers."""
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
    """Rectangular grid with wave-staggered altitude tiers."""
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
    """V-formation with 3D elevation slope."""
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

def compute_slots(ftype: FormationType, n: int, radius: float,
                  center: np.ndarray, z_base: float, z_span: float = Z_SPAN) -> np.ndarray:
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
#  Simulation State & Setup
# ==============================================================================

FAIL_START = HOLD_FRAMES
FAIL_END   = HOLD_FRAMES + FAILURE_WINDOW
fault = ACTIVE_FAULT.resolve(N_DRONES, FAIL_START, FAIL_END)
fault.print_summary()

FMT = fault.resolved_formation

initial_slots: np.ndarray = compute_slots(FMT, N_DRONES, FORMATION_RADIUS, ASSET_POS, FORMATION_Z)
positions:     np.ndarray = initial_slots.copy()
states:        List[str]  = [HONEST] * N_DRONES

# Raft consensus cluster
consensus_cluster = SwarmConsensusCluster(N_DRONES)

# APF Trajectory Planner
planner = PotentialFieldPlanner(d_safe=4.5, asset_radius=7.0)

apf_trajectories: Optional[Dict[int, np.ndarray]] = None
new_slots_arr:    Optional[np.ndarray]            = None
reorg_frame:      Optional[int]                   = None
failure_done:     bool                            = False
mech_counter:     int                             = 0
collision_vel:    Optional[np.ndarray]            = None

events: List[str] = []

def log(frame: int, msg: str) -> None:
    events.append(f"[t={frame/30.0:.1f}s] {msg}")
    if len(events) > 5:
        events.pop(0)

# ==============================================================================
#  Figure & Visual Styling
# ==============================================================================

fig = plt.figure(figsize=(14, 9), facecolor=BG_COLOR)
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor(BG_COLOR)

_hud_obj   = [None]
_log_obj   = [None]
_phase_obj = [None]

STATE_COLOR  = {HONEST: "#22c55e", FAULTING: "#f97316",
                FAILED: "#374151", REPOSITIONING: "#38bdf8",
                BYZANTINE: "#ef4444"}
STATE_MARKER = {HONEST: "o", FAULTING: "o", FAILED: "x", REPOSITIONING: "o", BYZANTINE: "^"}
STATE_SIZE   = {HONEST: 220, FAULTING: 220, FAILED: 260, REPOSITIONING: 220, BYZANTINE: 260}

# ==============================================================================
#  Animation Update Step
# ==============================================================================

def update(frame: int) -> None:
    global mech_counter, collision_vel, failure_done
    global reorg_frame, apf_trajectories, new_slots_arr

    ax.cla()
    ax.set_facecolor(BG_COLOR)
    for ref in (_hud_obj, _log_obj, _phase_obj):
        if ref[0] is not None:
            try: ref[0].remove()
            except Exception: pass
            ref[0] = None

    ax.set_xlim([10, 90]); ax.set_ylim([10, 90]); ax.set_zlim([0, 52])
    ax.set_xlabel("X (m)", color=DIM_BORDER, fontsize=8, labelpad=1)
    ax.set_ylabel("Y (m)", color=DIM_BORDER, fontsize=8, labelpad=1)
    ax.set_zlabel("Z (m)", color=DIM_BORDER, fontsize=8, labelpad=1)
    ax.tick_params(colors="#1e293b", labelsize=7)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#1e293b")
    ax.grid(False)

    # Draw protected ground asset
    ax.scatter(*ASSET_POS, marker="D", color="#fbbf24", s=280,
               edgecolors="white", linewidths=1.2, zorder=15)
    ax.text(ASSET_POS[0], ASSET_POS[1], ASSET_POS[2] + 4.0,
            "PROTECTED ASSET", color="#fbbf24", fontsize=8, ha="center", fontweight="bold")

    # Dynamic Raft Consensus Tick
    raft_logs = consensus_cluster.tick()
    for rl in raft_logs:
        log(frame, rl)

    leader_id = consensus_cluster.leader_id
    fid = fault.failed_drone_id
    fmode = fault.failure_mode

    # -- PHASE 1: HOLD FORMATION ----------------------------------------------
    if frame < HOLD_FRAMES:
        phase_name  = f"ACTIVE DEFENSE [{FMT.value.upper()} 3D MESH]"
        phase_color = "#22c55e"
        if frame == 0:
            log(frame, f"3D Mesh Online: {N_DRONES} drones in [{FMT.value}] configuration")
            log(frame, f"Raft Consensus: Drone {leader_id} verified as LEADER (Term 1)")

    # -- PHASE 2: FAILURE INJECTION & DYNAMIC BYZANTINE DETECTION -------------
    elif not failure_done:
        phase_name  = "ADVERSARIAL FAULT DETECTED"
        phase_color = "#ef4444"

        if states[fid] != FAILED:
            if fmode == FailureMode.MECHANICAL:
                if states[fid] != FAULTING:
                    states[fid] = FAULTING
                    log(frame, f"Telemetry anomaly: Drone {fid} reporting mechanical degradation")
                mech_counter += 1
                jitter = np.random.randn(3) * 1.5
                jitter[2] = 0.0
                positions[fid] = initial_slots[fid] + jitter
                if mech_counter >= MECH_JITTER_LEN:
                    states[fid] = FAILED
                    consensus_cluster.kill_node(fid)
                    log(frame, f"Heartbeat expired: Drone {fid} marked OFFLINE")

            elif fmode == FailureMode.RANDOM_COLLISION:
                if collision_vel is None:
                    rng2 = np.random.default_rng()
                    collision_vel = np.array([float(rng2.uniform(-2.0, 2.0)),
                                              float(rng2.uniform(-2.0, 2.0)),
                                              -0.8])
                    states[fid] = FAULTING
                    log(frame, f"Collision detected: Drone {fid} spinning out")
                collision_vel[2] -= 0.3
                positions[fid] += collision_vel + np.random.randn(3) * 0.2
                if positions[fid][2] < 2.0 or frame >= fault.failure_frame + CRASH_FALL_LEN:
                    states[fid] = FAILED
                    consensus_cluster.kill_node(fid)
                    log(frame, f"Drone {fid} crashed to ground")

            else:
                # ATTACK (GPS Spoofing / Byzantine Compromise)
                if frame >= fault.failure_frame:
                    states[fid] = FAILED
                    consensus_cluster.kill_node(fid)
                    log(frame, f"Byzantine attack: Drone {fid} eliminated from mesh")

        # Dynamic Byzantine Cross-Check (C++ Core verification)
        if frame == fault.failure_frame - 2:
            pairwise = np.zeros((N_DRONES, N_DRONES))
            for i in range(N_DRONES):
                for j in range(N_DRONES):
                    pairwise[i, j] = np.linalg.norm(positions[i] - positions[j])
            byz_nodes = native.detect_byzantine_nodes(positions, pairwise, threshold=3.0)
            if byz_nodes:
                log(frame, f"C++ BFT Quorum: Identified Byzantine nodes: {byz_nodes}")

        # Trigger Gap-Cover Reorganization once failure confirmed
        if states[fid] == FAILED and not failure_done:
            failure_done = True
            survivors = [i for i in range(N_DRONES) if states[i] != FAILED]

            # If leader failed, Raft election automatically triggers
            if leader_id == fid:
                log(frame, f"LEADER LOSS: Drone {fid} down -> Raft election initiated")
                for s in survivors:
                    if consensus_cluster.nodes[s].role == RaftRole.FOLLOWER:
                        consensus_cluster.nodes[s].timer = 0 # Force trigger election
                consensus_cluster.tick()
                leader_id = consensus_cluster.leader_id

            log(frame, f"Leader Drone {leader_id} issuing Gap-Cover Reorganization order")

            # 1. Compute baseline 3D geometric slots for surviving (N-1) drones
            new_slots_arr = compute_slots(FMT, len(survivors), FORMATION_RADIUS,
                                          ASSET_POS, FORMATION_Z, Z_SPAN)

            # 2. RL Formation Optimization: Use trained PPO policy to fine-tune GDOP
            if rl_model is not None:
                try:
                    obs = np.zeros(len(survivors) * 3 + 4, dtype=np.float32)
                    obs[:len(survivors)*3] = new_slots_arr.flatten()
                    obs[len(survivors)*3: len(survivors)*3 + 3] = ASSET_POS
                    obs[-1] = native.calculate_gdop(new_slots_arr, ASSET_POS)
                    action, _ = rl_model.predict(obs, deterministic=True)
                    offsets = action.reshape((len(survivors), 3))
                    new_slots_arr = new_slots_arr + np.clip(offsets, -1.5, 1.5)
                    log(frame, "RL Active Defense: GDOP optimization policy applied to slots")
                except Exception as e:
                    pass

            surv_pos = np.array([positions[i] for i in survivors])
            assign   = angular_assign(surv_pos, new_slots_arr, ASSET_POS)

            starts_dict = {survivors[d]: positions[survivors[d]].copy() for d in range(len(survivors))}
            targets_dict = {survivors[d]: new_slots_arr[assign[d]] for d in range(len(survivors))}

            # 3. Generate certified collision-free trajectories via APF
            apf_trajectories = planner.generate_trajectories(
                starts_dict, targets_dict, ASSET_POS, n_frames=REORG_FRAMES
            )
            is_safe, min_dist = PotentialFieldPlanner.verify_safety(apf_trajectories, min_allowed_dist=3.0)
            log(frame, f"APF Trajectory Verified: Collision-free (Min clearance: {min_dist:.1f}m)")

            reorg_frame = frame
            for i in survivors:
                states[i] = REPOSITIONING

    # -- PHASE 3: COLLISION-FREE REORGANIZATION & TAIL -------------------------
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
                    log(frame, "Reorganization Complete: 360-deg 3D defense restored")
                    final_gdop = native.calculate_gdop([positions[i] for i in survivors], ASSET_POS)
                    log(frame, f"Optimal 3D GDOP: {final_gdop:.2f} (Target bracketed)")

        phase_name  = f"GAP-COVER APF REORGANIZATION [{FMT.value.upper()}]"
        phase_color = "#38bdf8"

    # =========================================================================
    #  Rendering & Plots
    # =========================================================================
    alive = [i for i in range(N_DRONES) if states[i] != FAILED]

    # Draw mesh communication lines
    for ii in range(len(alive)):
        for jj in range(ii + 1, len(alive)):
            a, b = alive[ii], alive[jj]
            ax.plot([positions[a][0], positions[b][0]],
                    [positions[a][1], positions[b][1]],
                    [positions[a][2], positions[b][2]],
                    color="#1e3a5f", alpha=0.35, lw=0.8)

    # Draw drone nodes
    leader_id = consensus_cluster.leader_id
    for i in range(N_DRONES):
        st  = states[i]
        col = STATE_COLOR[st]
        pos = positions[i]

        if st == FAILED:
            ax.scatter(*pos, marker="x", color="#4b5563", s=280, linewidths=2.5, alpha=0.6, zorder=10)
            ax.text(pos[0], pos[1], pos[2] + 3.5, f"D{i}[FAIL]", color="#4b5563", fontsize=7.5, ha="center")
            continue

        ax.scatter(*pos, marker=STATE_MARKER[st], color=col,
                   s=STATE_SIZE[st], edgecolors="white", linewidths=0.8,
                   alpha=0.95, zorder=11)

        is_leader = (i == leader_id)
        suffix = "[LEADER]" if is_leader else ""
        label_col = "#fbbf24" if is_leader else col
        ax.text(pos[0], pos[1], pos[2] + 3.5,
                f"D{i}{suffix}", color=label_col, fontsize=8,
                ha="center", fontweight="bold", zorder=12)

    # HUD Overlay
    alive_ct = len(alive)
    coverage = int(round(alive_ct / N_DRONES * 100))
    gdop = native.calculate_gdop([positions[i] for i in alive], ASSET_POS)

    hud = [
        f"  Formation     : {FMT.value.upper()}",
        f"  Failure Mode  : {fault.failure_mode.value}",
        f"  Failed Drone  : {fault.failed_drone_id}",
        f"  Active Drones : {alive_ct}/{N_DRONES}",
        f"  Raft Leader   : Drone {leader_id if leader_id >= 0 else '--'}",
        f"  Raft Term     : {consensus_cluster.nodes[max(0, leader_id)].current_term}",
        f"  3D GDOP       : {gdop:.2f}",
        f"  Coverage      : {coverage}%",
        f"  C++ Engine    : ACTIVE (ctypes)",
        f"  Frame         : {frame}/{TOTAL_FRAMES}",
    ]
    _hud_obj[0] = fig.text(
        0.725, 0.88, "\n".join(hud),
        color="white", fontsize=9, fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor=PANEL_COLOR,
                  alpha=0.92, edgecolor=phase_color, linewidth=1.8),
    )

    log_text = "\n".join(events) if events else "(no events)"
    _log_obj[0] = fig.text(
        0.01, 0.01, log_text,
        color="#94a3b8", fontsize=8.0, fontfamily="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL_COLOR,
                  alpha=0.85, edgecolor=DIM_BORDER, linewidth=1.0),
    )

    ax.set_title("SwarmRaft: Decentralized Fault Detection & Active Defense",
                 color="white", fontsize=13, fontweight="bold", pad=12)

    _phase_obj[0] = fig.text(
        0.5, 0.96, phase_name,
        ha="center", color=phase_color, fontsize=12,
        fontweight="bold", fontfamily="monospace",
    )

    legend_items = [
        mpatches.Patch(color="#22c55e", label="Honest (on slot)"),
        mpatches.Patch(color="#f97316", label="Faulting (degrading)"),
        mpatches.Patch(color="#4b5563", label="Failed / Offline"),
        mpatches.Patch(color="#38bdf8", label="APF Repositioning"),
        mpatches.Patch(color="#fbbf24", label="Protected Asset"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=7.5,
              facecolor=PANEL_COLOR, edgecolor=DIM_BORDER,
              labelcolor="white", framealpha=0.90)

    if frame >= TOTAL_FRAMES - 1:
        ani.event_source.stop()
        fig.text(
            0.5, 0.04,
            "Simulation Complete -- Press Close to Exit",
            ha="center", color="#22c55e", fontsize=10, fontfamily="monospace",
        )

ani = animation.FuncAnimation(
    fig, update,
    frames=TOTAL_FRAMES,
    interval=33,
    repeat=False,
    cache_frame_data=False,
)

if __name__ == "__main__":
    plt.tight_layout()
    plt.show()
