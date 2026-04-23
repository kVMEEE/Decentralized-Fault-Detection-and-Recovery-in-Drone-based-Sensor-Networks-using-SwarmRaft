# -*- coding: utf-8 -*-
# =============================================================================
#  SwarmRaft -- RL Active Defense Formation Simulation
#  ---------------------------------------------------------------------------
#  Shows the RL active-defense layer only (Phase 2 of the full system):
#    1. Drones hold a configurable perimeter formation around the asset.
#    2. One drone fails (mode / target set in fault_injector.py).
#    3. SwarmRaft Raft consensus detects the failure.
#    4. Remaining drones reorganise into a new formation that covers the
#       compromised slot -- same shape, N-1 evenly distributed slots.
#
#  Formation type and failure scenario are both resolved at runtime by
#  fault_injector.py.  Nothing in this file is hardcoded.
#
#  Run:   python swarm_sim.py
#  Edit:  fault_injector.py  (change mode, formation, target drone etc.)
# =============================================================================

import sys
import os
import math
from typing import Dict, List, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

matplotlib.rcParams["font.family"] = "DejaVu Sans"

# -- import fault config -------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fault_injector import ACTIVE_FAULT, FailureMode, FormationType

# ==============================================================================
#  Simulation constants
# ==============================================================================

N_DRONES         = 8
FORMATION_RADIUS = 30.0
ASSET_POS        = np.array([50.0, 50.0, 22.0])
FORMATION_Z      = 22.0

HOLD_FRAMES      = 60    # hold formation before failure window opens
FAILURE_WINDOW   = 40    # window of frames in which failure can fire
REORG_FRAMES     = 100   # interpolation length for repositioning
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

# ==============================================================================
#  Formation slot functions
#  All return an (n, 3) ndarray.
#  'radius' controls the spread; 'center' is the asset position (perimeter hub).
# ==============================================================================

def ring_slots(n: int, radius: float, center: np.ndarray, z: float) -> np.ndarray:
    """Evenly spaced circle."""
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
        np.full(n, z),
    ], axis=1)


def diamond_slots(n: int, radius: float, center: np.ndarray, z: float) -> np.ndarray:
    """Drones distributed along the 4 edges of a rotated square (diamond).
    Cardinal corners at N / E / S / W at full radius."""
    corners = np.array([
        [center[0],           center[1] + radius],   # North
        [center[0] + radius,  center[1]           ],  # East
        [center[0],           center[1] - radius],   # South
        [center[0] - radius,  center[1]           ],  # West
    ])
    t_vals = np.linspace(0, 1, n, endpoint=False)
    pts = []
    for t in t_vals:
        edge = int(t * 4) % 4
        et   = (t * 4) - int(t * 4)
        p    = corners[edge] + et * (corners[(edge + 1) % 4] - corners[edge])
        pts.append([p[0], p[1], z])
    return np.array(pts)


def grid_slots(n: int, radius: float, center: np.ndarray, z: float) -> np.ndarray:
    """Rectangular 4-column grid. Spaced to avoid placing a drone at the
    asset center."""
    cols = max(2, math.ceil(math.sqrt(n * 2)))   # prefer wide layout
    rows = math.ceil(n / cols)
    # Use (cols+1) intervals so drones sit between the edges, not on them
    dx = 2 * radius / (cols + 1)
    dy = 2 * radius / (rows + 1)
    ox = center[0] - radius
    oy = center[1] - radius
    pts = []
    for row in range(rows):
        for col in range(cols):
            if len(pts) >= n:
                break
            pts.append([ox + (col + 1) * dx,
                         oy + (row + 1) * dy,
                         z])
    return np.array(pts[:n])


def v_slots(n: int, radius: float, center: np.ndarray, z: float) -> np.ndarray:
    """V-formation: 1 tip at front, equal arms extending backward at ~55 deg."""
    left_arm  = (n - 1) // 2
    right_arm = n - 1 - left_arm    # handles even n (right gets 1 extra)
    half_ang  = math.radians(55)
    front_y   = center[1] + radius * 0.7
    step      = radius / max(max(left_arm, right_arm), 1)

    pts = [[center[0], front_y, z]]    # tip

    for k in range(1, left_arm + 1):
        d = step * k
        pts.append([center[0] - d * math.sin(half_ang),
                     front_y  - d * math.cos(half_ang),
                     z])

    for k in range(1, right_arm + 1):
        d = step * k
        pts.append([center[0] + d * math.sin(half_ang),
                     front_y  - d * math.cos(half_ang),
                     z])

    return np.array(pts)


def compute_slots(ftype: FormationType, n: int, radius: float,
                  center: np.ndarray, z: float) -> np.ndarray:
    """Dispatch to the correct slot function based on formation type."""
    fn = {
        FormationType.RING:    ring_slots,
        FormationType.DIAMOND: diamond_slots,
        FormationType.GRID:    grid_slots,
        FormationType.V_SHAPE: v_slots,
    }.get(ftype, ring_slots)
    return fn(n, radius, center, z)


# ==============================================================================
#  Helpers
# ==============================================================================

def angular_assign(drone_pos: np.ndarray, slot_pos: np.ndarray,
                   center: np.ndarray) -> List[int]:
    """Assign drones to slots by angular order to minimise crossing paths."""
    def angle_of(p):
        return math.atan2(p[1] - center[1], p[0] - center[0])

    n = len(drone_pos)
    d_order = sorted(range(n), key=lambda i: angle_of(drone_pos[i]))
    s_order = sorted(range(n), key=lambda i: angle_of(slot_pos[i]))

    result = [0] * n
    for rank, d_idx in enumerate(d_order):
        result[d_idx] = s_order[rank]
    return result


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def calc_gdop(positions: List[np.ndarray], target: np.ndarray) -> float:
    H = []
    for pos in positions:
        diff = target - np.asarray(pos)
        d = float(np.linalg.norm(diff))
        if d < 1e-6:
            d = 1e-6
        uv = diff / d
        H.append([uv[0], uv[1], uv[2], 1.0])
    if len(H) < 4:
        return 99.9
    Hm = np.array(H)
    try:
        return float(min(np.sqrt(np.trace(np.linalg.inv(Hm.T @ Hm))), 99.9))
    except np.linalg.LinAlgError:
        return 99.9


def draw_outline(ax, pts: np.ndarray, color: str, alpha: float,
                 lw: float = 1.2, style: str = "--") -> None:
    """Connect formation points in angular order to show the outlined shape."""
    if len(pts) < 2:
        return
    order = np.argsort(np.arctan2(pts[:, 1] - ASSET_POS[1],
                                   pts[:, 0] - ASSET_POS[0]))
    closed = np.vstack([pts[order], pts[order[0:1]]])
    ax.plot(closed[:, 0], closed[:, 1], closed[:, 2],
            style, color=color, alpha=alpha, lw=lw, zorder=0)


# ==============================================================================
#  Resolve fault
# ==============================================================================

FAIL_START = HOLD_FRAMES
FAIL_END   = HOLD_FRAMES + FAILURE_WINDOW

fault = ACTIVE_FAULT.resolve(N_DRONES, FAIL_START, FAIL_END)
fault.print_summary()

FMT = fault.resolved_formation   # short alias

# ==============================================================================
#  Simulation state
# ==============================================================================

initial_slots: np.ndarray    = compute_slots(FMT, N_DRONES, FORMATION_RADIUS,
                                             ASSET_POS, FORMATION_Z)
positions:     np.ndarray    = initial_slots.copy()
states:        List[str]     = [HONEST] * N_DRONES
leader_id:     int           = 0

reorg_start:   Dict[int, np.ndarray] = {}
reorg_target:  Dict[int, np.ndarray] = {}
new_slots_arr: Optional[np.ndarray]  = None
reorg_frame:   Optional[int]         = None
failure_done:  bool                  = False

mech_counter:    int                   = 0
collision_vel:   Optional[np.ndarray] = None

events: List[str] = []

def log(frame: int, msg: str) -> None:
    events.append(f"[t={frame/30.0:.1f}s]  {msg}")
    if len(events) > 5:
        events.pop(0)

# ==============================================================================
#  Figure
# ==============================================================================

fig = plt.figure(figsize=(14, 9), facecolor=BG_COLOR)
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor(BG_COLOR)

_hud_obj   = [None]
_log_obj   = [None]
_phase_obj = [None]

STATE_COLOR  = {HONEST: "#22c55e", FAULTING: "#f97316",
                FAILED: "#374151", REPOSITIONING: "#38bdf8"}
STATE_MARKER = {HONEST: "o", FAULTING: "o", FAILED: "x", REPOSITIONING: "o"}
STATE_SIZE   = {HONEST: 220, FAULTING: 220, FAILED: 260, REPOSITIONING: 220}

# ==============================================================================
#  Animation update
# ==============================================================================

def update(frame: int) -> None:
    global mech_counter, collision_vel, failure_done, leader_id
    global reorg_frame, reorg_start, reorg_target, new_slots_arr

    # -- clear frame ----------------------------------------------------------
    ax.cla()
    ax.set_facecolor(BG_COLOR)
    for ref in (_hud_obj, _log_obj, _phase_obj):
        if ref[0] is not None:
            try:
                ref[0].remove()
            except Exception:
                pass
            ref[0] = None

    # -- axis style -----------------------------------------------------------
    ax.set_xlim([10, 90]); ax.set_ylim([10, 90]); ax.set_zlim([0, 52])
    ax.set_xlabel("X (m)", color=DIM_BORDER, fontsize=8, labelpad=1)
    ax.set_ylabel("Y (m)", color=DIM_BORDER, fontsize=8, labelpad=1)
    ax.set_zlabel("Z (m)", color=DIM_BORDER, fontsize=8, labelpad=1)
    ax.tick_params(colors="#1e293b", labelsize=7)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#1e293b")
    ax.grid(False)

    # -- asset ----------------------------------------------------------------
    ax.scatter(*ASSET_POS, marker="D", color="#fbbf24", s=280,
               edgecolors="white", linewidths=1.2, zorder=15)
    ax.text(ASSET_POS[0], ASSET_POS[1], ASSET_POS[2] + 4.5,
            "ASSET", color="#fbbf24", fontsize=8, ha="center", fontweight="bold")

    # =========================================================================
    #  State machine
    # =========================================================================

    fid   = fault.failed_drone_id
    fmode = fault.failure_mode

    # -- HOLD: show initial formation -----------------------------------------
    if frame < HOLD_FRAMES:
        phase_name  = f"ACTIVE DEFENSE  [{FMT.value.upper()} FORMATION]"
        phase_color = "#22c55e"
        if frame == 0:
            log(frame, f"Formation ACTIVE -- {N_DRONES} drones on perimeter [{FMT.value}]")
            log(frame, f"Drone {leader_id} is Raft LEADER")

    # -- FAILURE injection window ---------------------------------------------
    elif not failure_done:
        phase_name  = "FAILURE DETECTED"
        phase_color = "#ef4444"

        if states[fid] != FAILED:

            # MECHANICAL -------------------------------------------------------
            if fmode == FailureMode.MECHANICAL:
                if states[fid] != FAULTING:
                    states[fid] = FAULTING
                    log(frame, f"Drone {fid} -- MECHANICAL FAULT (degrading)")
                mech_counter += 1
                jitter = np.random.randn(3) * 1.6
                jitter[2] = 0.0
                positions[fid] = initial_slots[fid] + jitter
                if mech_counter >= MECH_JITTER_LEN:
                    states[fid] = FAILED
                    log(frame, f"Drone {fid} -- SYSTEM FAILURE")

            # COLLISION --------------------------------------------------------
            elif fmode == FailureMode.RANDOM_COLLISION:
                if collision_vel is None:
                    rng2 = np.random.default_rng()
                    collision_vel = np.array([float(rng2.uniform(-2.2, 2.2)),
                                              float(rng2.uniform(-2.2, 2.2)),
                                              -0.9])
                    states[fid] = FAULTING
                    log(frame, f"Drone {fid} -- COLLISION (spinning out)")
                collision_vel[2] -= 0.32
                positions[fid] += collision_vel + np.random.randn(3) * 0.3
                if (positions[fid][2] < 2.0
                        or frame >= fault.failure_frame + CRASH_FALL_LEN):
                    states[fid] = FAILED
                    log(frame, f"Drone {fid} -- CRASHED")

            # ATTACK / TARGETED ------------------------------------------------
            else:
                if frame >= fault.failure_frame:
                    states[fid] = FAILED
                    log(frame, f"Drone {fid} -- DESTROYED (ATTACK)")

        # -- Trigger reorg once failure confirmed ----------------------------
        if states[fid] == FAILED and not failure_done:
            failure_done = True

            survivors = [i for i in range(N_DRONES) if states[i] != FAILED]

            if leader_id == fid:
                leader_id = survivors[0] if survivors else -1
                log(frame, f"Raft election -- Drone {leader_id} elected LEADER")
            else:
                log(frame, f"Drone {leader_id} (leader) initiating gap-cover")

            log(frame, f"Gap-cover order broadcast -- {len(survivors)} drones respond")

            # New slots: same formation type, N-1 slots
            new_slots_arr = compute_slots(FMT, len(survivors), FORMATION_RADIUS,
                                          ASSET_POS, FORMATION_Z)
            surv_pos = np.array([positions[i] for i in survivors])
            assign   = angular_assign(surv_pos, new_slots_arr, ASSET_POS)

            reorg_start  = {survivors[d]: positions[survivors[d]].copy()
                            for d in range(len(survivors))}
            reorg_target = {survivors[d]: new_slots_arr[assign[d]]
                            for d in range(len(survivors))}
            reorg_frame  = frame

            for i in survivors:
                states[i] = REPOSITIONING

    # -- REORG + TAIL ---------------------------------------------------------
    else:
        if reorg_frame is not None:
            elapsed = frame - reorg_frame
            t_raw   = elapsed / REORG_FRAMES
            t       = smoothstep(t_raw)

            survivors = [i for i in range(N_DRONES) if states[i] != FAILED]
            for i in survivors:
                if i in reorg_target:
                    positions[i] = reorg_start[i] + (reorg_target[i] - reorg_start[i]) * t

            if t_raw >= 1.0:
                for i in survivors:
                    if states[i] == REPOSITIONING:
                        states[i] = HONEST
                if frame == reorg_frame + REORG_FRAMES:
                    log(frame, "Reorg complete -- 360 deg coverage maintained")
                    alive_pos = [positions[i] for i in survivors]
                    log(frame, f"Final GDOP: {calc_gdop(alive_pos, ASSET_POS):.2f}")

        phase_name  = f"GAP-COVER REORG  [{FMT.value.upper()}]"
        phase_color = "#38bdf8"

    # =========================================================================
    #  Drawing
    # =========================================================================

    alive = [i for i in range(N_DRONES) if states[i] != FAILED]

    # -- formation outline (dynamic, morphs during reorg) --------------------
    if len(alive) >= 2:
        alive_pos_arr = np.array([positions[i] for i in alive])
        draw_outline(ax, alive_pos_arr, DIM_BORDER, 0.45, lw=1.3)

    # -- target formation outline (during reorg) ------------------------------
    if reorg_frame is not None and new_slots_arr is not None:
        elapsed_d = frame - reorg_frame
        if elapsed_d / REORG_FRAMES < 1.0:
            draw_outline(ax, new_slots_arr, "#38bdf8", 0.20, lw=0.9, style="-")

    # -- mesh lines -----------------------------------------------------------
    for ii in range(len(alive)):
        for jj in range(ii + 1, len(alive)):
            a, b = alive[ii], alive[jj]
            ax.plot([positions[a][0], positions[b][0]],
                    [positions[a][1], positions[b][1]],
                    [positions[a][2], positions[b][2]],
                    color="#1e3a5f", alpha=0.28, lw=0.7)

    # -- reorg direction guides -----------------------------------------------
    if reorg_frame is not None:
        elapsed_g = frame - reorg_frame
        if elapsed_g / REORG_FRAMES < 1.0:
            for i, tgt in reorg_target.items():
                if states[i] == REPOSITIONING:
                    ax.plot([positions[i][0], tgt[0]],
                            [positions[i][1], tgt[1]],
                            [positions[i][2], tgt[2]],
                            color="#38bdf8", alpha=0.35, lw=1.0, linestyle="--")

    # -- drones ---------------------------------------------------------------
    for i in range(N_DRONES):
        st  = states[i]
        col = STATE_COLOR[st]
        pos = positions[i]

        if st == FAILED:
            ax.scatter(*pos, marker="x", color="#374151",
                       s=280, linewidths=2.5, alpha=0.55, zorder=10)
            ax.text(pos[0], pos[1], pos[2] + 3.5,
                    f"D{i}[FAIL]", color="#374151", fontsize=7.5, ha="center")
            continue

        ax.scatter(*pos, marker=STATE_MARKER[st], color=col,
                   s=STATE_SIZE[st], edgecolors="white", linewidths=0.7,
                   alpha=0.92, zorder=11)

        suffix = "[L]" if i == leader_id else ""
        ax.text(pos[0], pos[1], pos[2] + 3.5,
                f"D{i}{suffix}", color=col, fontsize=8,
                ha="center", fontweight="bold", zorder=12)

    # =========================================================================
    #  HUD
    # =========================================================================

    alive_ct = len(alive)
    coverage = int(round(alive_ct / N_DRONES * 100))
    gdop     = calc_gdop([positions[i] for i in alive], ASSET_POS)

    hud = [
        f"  Formation     : {FMT.value}",
        f"  Failure Mode  : {fault.failure_mode.value}",
        f"  Failed Drone  : {fault.failed_drone_id}",
        f"  Active Drones : {alive_ct}/{N_DRONES}",
        f"  Leader        : Drone {leader_id if leader_id >= 0 else '--'}",
        f"  GDOP          : {gdop:.2f}",
        f"  Coverage      : {coverage}%",
        f"  Frame         : {frame}/{TOTAL_FRAMES}",
    ]
    _hud_obj[0] = fig.text(
        0.735, 0.88, "\n".join(hud),
        color="white", fontsize=9, fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor=PANEL_COLOR,
                  alpha=0.90, edgecolor=phase_color, linewidth=1.8),
    )

    log_text = "\n".join(events) if events else "(no events)"
    _log_obj[0] = fig.text(
        0.01, 0.01, log_text,
        color="#94a3b8", fontsize=7.8, fontfamily="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL_COLOR,
                  alpha=0.82, edgecolor=DIM_BORDER, linewidth=1.0),
    )

    ax.set_title("SwarmRaft -- RL Active Defense Simulation",
                 color="white", fontsize=13, fontweight="bold", pad=12)

    _phase_obj[0] = fig.text(
        0.5, 0.96, phase_name,
        ha="center", color=phase_color, fontsize=12,
        fontweight="bold", fontfamily="monospace",
    )

    legend_items = [
        mpatches.Patch(color="#22c55e", label="Honest (on slot)"),
        mpatches.Patch(color="#f97316", label="Faulting (pre-failure)"),
        mpatches.Patch(color="#374151", label="Failed"),
        mpatches.Patch(color="#38bdf8", label="Repositioning"),
        mpatches.Patch(color="#fbbf24", label="Protected Asset"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=7.5,
              facecolor=PANEL_COLOR, edgecolor=DIM_BORDER,
              labelcolor="white", framealpha=0.88)

    # -- pause on last frame -------------------------------------------------
    if frame >= TOTAL_FRAMES - 1:
        ani.event_source.stop()
        # Overlay completion banner
        fig.text(
            0.5, 0.50,
            "SIMULATION COMPLETE",
            ha="center", va="center", color="#22c55e",
            fontsize=18, fontweight="bold", fontfamily="monospace",
            alpha=0.0,   # invisible placeholder so bbox sizes properly
        )
        fig.text(
            0.5, 0.04,
            "Simulation paused  --  close window to exit",
            ha="center", color="#64748b", fontsize=9, fontfamily="monospace",
        )


# ==============================================================================
#  Run
# ==============================================================================

ani = animation.FuncAnimation(
    fig, update,
    frames=TOTAL_FRAMES,
    interval=33,
    repeat=False,           # do not loop after last frame
    cache_frame_data=False,
)

if __name__ == "__main__":
    plt.tight_layout()
    plt.show()
