import os
import sys
import ctypes
import numpy as np
from typing import List, Tuple, Optional

# Locate DLL in cpp_mesh
_DIR = os.path.dirname(os.path.abspath(__file__))
_DLL_PATH = os.path.join(os.path.dirname(_DIR), "cpp_mesh", "swarmraft_core.dll")

_lib = None
if os.path.exists(_DLL_PATH):
    try:
        _lib = ctypes.CDLL(_DLL_PATH)
        # Define argtypes and restype
        _lib.solve_multilateration_3d.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_double
        ]
        _lib.solve_multilateration_3d.restype = ctypes.c_int

        _lib.detect_byzantine_nodes.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_int)
        ]
        _lib.detect_byzantine_nodes.restype = ctypes.c_int

        _lib.calculate_gdop_3d.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int
        ]
        _lib.calculate_gdop_3d.restype = ctypes.c_double
    except Exception as e:
        print(f"[SwarmRaft Native] Warning: Failed to load C++ DLL: {e}. Falling back to Python math.", file=sys.stderr)
        _lib = None


def solve_multilateration(anchors: np.ndarray, distances: np.ndarray, initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
    """
    3D Multilateration Levenberg-Marquardt solver.
    Uses C++ compiled engine via ctypes if available, else numpy fallback.
    """
    n = len(anchors)
    if n < 4:
        return initial_guess.copy() if initial_guess is not None else np.zeros(3)

    guess = initial_guess.copy() if initial_guess is not None else np.mean(anchors, axis=0)

    if _lib is not None:
        c_anchors = (ctypes.c_double * (n * 3))(*anchors.flatten())
        c_dist = (ctypes.c_double * n)(*distances)
        c_out = (ctypes.c_double * 3)(*guess)
        res = _lib.solve_multilateration_3d(c_anchors, c_dist, n, c_out, 50, 1e-4)
        if res == 0:
            return np.array([c_out[0], c_out[1], c_out[2]], dtype=np.float64)

    # Python Fallback
    pos = guess.astype(np.float64).copy()
    for _ in range(50):
        diffs = pos - anchors
        est_dists = np.linalg.norm(diffs, axis=1)
        est_dists = np.maximum(est_dists, 1e-6)
        r = est_dists - distances
        J = diffs / est_dists[:, np.newaxis]
        JtJ = J.T @ J + 1e-3 * np.eye(3)
        try:
            delta = -np.linalg.solve(JtJ, J.T @ r)
            pos += delta
            if np.linalg.norm(delta) < 1e-4:
                break
        except np.linalg.LinAlgError:
            break
    return pos


def detect_byzantine_nodes(reported_coords: np.ndarray, pairwise_distances: np.ndarray, threshold: float = 2.5) -> List[int]:
    """
    Detects malicious/spoofed nodes dynamically via pairwise cross-ranging consistency check.
    Returns list of Byzantine drone IDs.
    """
    n = len(reported_coords)
    if n < 3:
        return []

    if _lib is not None:
        c_coords = (ctypes.c_double * (n * 3))(*reported_coords.flatten())
        c_dist = (ctypes.c_double * (n * n))(*pairwise_distances.flatten())
        c_out = (ctypes.c_int * n)()
        res = _lib.detect_byzantine_nodes(c_coords, c_dist, n, threshold, c_out)
        if res == 0:
            return [i for i in range(n) if c_out[i] == 1]

    # Python Fallback
    suspect_votes = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            geom_d = np.linalg.norm(reported_coords[i] - reported_coords[j])
            meas_d = pairwise_distances[i, j]
            if abs(geom_d - meas_d) > threshold:
                suspect_votes[i] += 1
                suspect_votes[j] += 1

    quorum = max(1, (n - 1) // 2)
    return [i for i in range(n) if suspect_votes[i] >= quorum]


def calculate_gdop(positions: np.ndarray, target: np.ndarray) -> float:
    """
    Computes 3D GDOP via C++ engine if available, with robust singular check.
    """
    n = len(positions)
    if n < 4:
        return 99.9

    if _lib is not None:
        c_pos = (ctypes.c_double * (n * 3))(*np.asarray(positions, dtype=np.float64).flatten())
        c_target = (ctypes.c_double * 3)(*np.asarray(target, dtype=np.float64))
        val = _lib.calculate_gdop_3d(c_pos, c_target, n)
        return float(val)

    # Python Fallback
    H = []
    for p in positions:
        diff = target - p
        d = max(float(np.linalg.norm(diff)), 1e-6)
        H.append([diff[0]/d, diff[1]/d, diff[2]/d, 1.0])
    Hm = np.array(H)
    try:
        Q = np.linalg.inv(Hm.T @ Hm)
        trace = np.trace(Q)
        return float(min(np.sqrt(trace), 99.9)) if trace > 0 else 99.9
    except np.linalg.LinAlgError:
        return 99.9
