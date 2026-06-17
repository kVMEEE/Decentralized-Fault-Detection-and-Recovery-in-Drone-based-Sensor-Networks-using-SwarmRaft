import sys
import os
import math
import asyncio
from typing import Dict, List, Optional
import numpy as np

# -- import fault config -------------------------------------------------------
# Ensure we can import fault_injector from parent directory or same directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from python_layer.fault_injector import ACTIVE_FAULT, FailureMode, FormationType
except ImportError:
    # If copied directly, try local import
    from fault_injector import ACTIVE_FAULT, FailureMode, FormationType

# ==============================================================================
#  Simulation constants
# ==============================================================================
N_DRONES         = 8
FORMATION_RADIUS = 30.0
ASSET_POS        = np.array([50.0, 50.0, 22.0])
FORMATION_Z      = 22.0

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

# ==============================================================================
#  Formation slot functions
# ==============================================================================
def ring_slots(n: int, radius: float, center: np.ndarray, z: float) -> np.ndarray:
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
        np.full(n, z),
    ], axis=1)

def diamond_slots(n: int, radius: float, center: np.ndarray, z: float) -> np.ndarray:
    corners = np.array([
        [center[0],           center[1] + radius],
        [center[0] + radius,  center[1]           ],
        [center[0],           center[1] - radius],
        [center[0] - radius,  center[1]           ],
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
    cols = max(2, math.ceil(math.sqrt(n * 2)))
    rows = math.ceil(n / cols)
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
    left_arm  = (n - 1) // 2
    right_arm = n - 1 - left_arm
    half_ang  = math.radians(55)
    front_y   = center[1] + radius * 0.7
    step      = radius / max(max(left_arm, right_arm), 1)
    pts = [[center[0], front_y, z]]
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

def compute_slots(ftype: FormationType, n: int, radius: float, center: np.ndarray, z: float) -> np.ndarray:
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

def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)

def calc_gdop(positions: List[np.ndarray], target: np.ndarray) -> float:
    H = []
    for pos in positions:
        diff = target - np.asarray(pos)
        d = float(np.linalg.norm(diff))
        if d < 1e-6: d = 1e-6
        uv = diff / d
        H.append([uv[0], uv[1], uv[2], 1.0])
    if len(H) < 4: return 99.9
    Hm = np.array(H)
    try:
        return float(min(np.sqrt(np.trace(np.linalg.inv(Hm.T @ Hm))), 99.9))
    except np.linalg.LinAlgError:
        return 99.9

# ==============================================================================
#  Generator logic
# ==============================================================================
async def run_simulation(fps=30):
    """
    Async generator that runs the simulation frame by frame and yields the state.
    """
    FAIL_START = HOLD_FRAMES
    FAIL_END   = HOLD_FRAMES + FAILURE_WINDOW
    fault = ACTIVE_FAULT.resolve(N_DRONES, FAIL_START, FAIL_END)
    FMT = fault.resolved_formation

    initial_slots = compute_slots(FMT, N_DRONES, FORMATION_RADIUS, ASSET_POS, FORMATION_Z)
    positions = initial_slots.copy()
    states = [HONEST] * N_DRONES
    leader_id = 0

    reorg_start = {}
    reorg_target = {}
    new_slots_arr = None
    reorg_frame = None
    failure_done = False
    mech_counter = 0
    collision_vel = None
    events = []

    def log(msg: str):
        events.append(msg)
        if len(events) > 5: events.pop(0)

    for frame in range(TOTAL_FRAMES):
        fid = fault.failed_drone_id
        fmode = fault.failure_mode

        phase_name = ""
        
        # -- HOLD: show initial formation -----------------------------------------
        if frame < HOLD_FRAMES:
            phase_name = f"ACTIVE DEFENSE [{FMT.value.upper()} FORMATION]"
            if frame == 0:
                log(f"Formation ACTIVE -- {N_DRONES} drones on perimeter [{FMT.value}]")
                log(f"Drone {leader_id} is Raft LEADER")

        # -- FAILURE injection window ---------------------------------------------
        elif not failure_done:
            phase_name = "FAILURE DETECTED"
            if states[fid] != FAILED:
                # MECHANICAL
                if fmode == FailureMode.MECHANICAL:
                    if states[fid] != FAULTING:
                        states[fid] = FAULTING
                        log(f"Drone {fid} -- MECHANICAL FAULT (degrading)")
                    mech_counter += 1
                    jitter = np.random.randn(3) * 1.6
                    jitter[2] = 0.0
                    positions[fid] = initial_slots[fid] + jitter
                    if mech_counter >= MECH_JITTER_LEN:
                        states[fid] = FAILED
                        log(f"Drone {fid} -- SYSTEM FAILURE")
                # COLLISION
                elif fmode == FailureMode.RANDOM_COLLISION:
                    if collision_vel is None:
                        rng2 = np.random.default_rng()
                        collision_vel = np.array([float(rng2.uniform(-2.2, 2.2)),
                                                  float(rng2.uniform(-2.2, 2.2)),
                                                  -0.9])
                        states[fid] = FAULTING
                        log(f"Drone {fid} -- COLLISION (spinning out)")
                    collision_vel[2] -= 0.32
                    positions[fid] += collision_vel + np.random.randn(3) * 0.3
                    if (positions[fid][2] < 2.0 or frame >= fault.failure_frame + CRASH_FALL_LEN):
                        states[fid] = FAILED
                        log(f"Drone {fid} -- CRASHED")
                # ATTACK
                else:
                    if frame >= fault.failure_frame:
                        states[fid] = FAILED
                        log(f"Drone {fid} -- DESTROYED (ATTACK)")

            # -- Trigger reorg once failure confirmed ----------------------------
            if states[fid] == FAILED and not failure_done:
                failure_done = True
                survivors = [i for i in range(N_DRONES) if states[i] != FAILED]
                if leader_id == fid:
                    leader_id = survivors[0] if survivors else -1
                    log(f"Raft election -- Drone {leader_id} elected LEADER")
                else:
                    log(f"Drone {leader_id} (leader) initiating gap-cover")
                log(f"Gap-cover order broadcast -- {len(survivors)} drones respond")

                new_slots_arr = compute_slots(FMT, len(survivors), FORMATION_RADIUS, ASSET_POS, FORMATION_Z)
                surv_pos = np.array([positions[i] for i in survivors])
                assign = angular_assign(surv_pos, new_slots_arr, ASSET_POS)

                reorg_start = {survivors[d]: positions[survivors[d]].copy() for d in range(len(survivors))}
                reorg_target = {survivors[d]: new_slots_arr[assign[d]] for d in range(len(survivors))}
                reorg_frame = frame

                for i in survivors: states[i] = REPOSITIONING

        # -- REORG + TAIL ---------------------------------------------------------
        else:
            if reorg_frame is not None:
                elapsed = frame - reorg_frame
                t_raw = elapsed / REORG_FRAMES
                t = smoothstep(t_raw)
                survivors = [i for i in range(N_DRONES) if states[i] != FAILED]
                for i in survivors:
                    if i in reorg_target:
                        positions[i] = reorg_start[i] + (reorg_target[i] - reorg_start[i]) * t
                if t_raw >= 1.0:
                    for i in survivors:
                        if states[i] == REPOSITIONING: states[i] = HONEST
                    if frame == reorg_frame + REORG_FRAMES:
                        log("Reorg complete -- 360 deg coverage maintained")
                        alive_pos = [positions[i] for i in survivors]
                        log(f"Final GDOP: {calc_gdop(alive_pos, ASSET_POS):.2f}")
            phase_name = f"GAP-COVER REORG [{FMT.value.upper()}]"

        # Calculate metrics
        alive = [i for i in range(N_DRONES) if states[i] != FAILED]
        alive_ct = len(alive)
        coverage = int(round(alive_ct / N_DRONES * 100))
        gdop = calc_gdop([positions[i] for i in alive], ASSET_POS)

        # Yield frame state
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
            },
            "logs": events.copy()
        }
        
        yield state_dict
        await asyncio.sleep(1 / fps)

    # Yield one final message to indicate completion
    yield {"status": "complete"}
