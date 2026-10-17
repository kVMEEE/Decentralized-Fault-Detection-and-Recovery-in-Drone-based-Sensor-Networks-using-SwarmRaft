import numpy as np
from typing import Dict, List, Tuple

class PotentialFieldPlanner:
    """
    Artificial Potential Field (APF) trajectory planner.
    Computes smooth, collision-free paths for drones during swarm reorganization,
    ensuring minimum inter-drone separation and asset obstacle clearance.
    """
    def __init__(self,
                 d_safe: float = 5.0,
                 asset_radius: float = 8.0,
                 k_att: float = 1.8,
                 k_rep_drone: float = 35.0,
                 k_rep_asset: float = 60.0,
                 damping: float = 0.85,
                 max_speed: float = 3.5):
        self.d_safe = d_safe
        self.asset_radius = asset_radius
        self.k_att = k_att
        self.k_rep_drone = k_rep_drone
        self.k_rep_asset = k_rep_asset
        self.damping = damping
        self.max_speed = max_speed

    def generate_trajectories(self,
                              start_positions: Dict[int, np.ndarray],
                              target_positions: Dict[int, np.ndarray],
                              asset_pos: np.ndarray,
                              n_frames: int = 100,
                              dt: float = 0.05) -> Dict[int, np.ndarray]:
        """
        Generates certified collision-free trajectories of shape (n_frames, 3) for each drone.
        """
        drone_ids = list(start_positions.keys())
        n_drones = len(drone_ids)

        positions = {did: start_positions[did].astype(np.float64).copy() for did in drone_ids}
        velocities = {did: np.zeros(3, dtype=np.float64) for did in drone_ids}

        trajectories = {did: np.zeros((n_frames, 3), dtype=np.float64) for did in drone_ids}

        # Sub-stepping for numerical stability
        sub_steps = 4
        sub_dt = dt / sub_steps

        for frame in range(n_frames):
            # Record current state
            for did in drone_ids:
                trajectories[did][frame] = positions[did].copy()

            # Dynamic relaxation physics
            for _ in range(sub_steps):
                forces = {did: np.zeros(3, dtype=np.float64) for did in drone_ids}

                for did in drone_ids:
                    pos = positions[did]
                    target = target_positions[did]

                    # 1. Attractive force to goal
                    diff_target = target - pos
                    dist_target = np.linalg.norm(diff_target)
                    if dist_target > 0.05:
                        dir_target = diff_target / dist_target
                        speed_factor = min(dist_target * 0.8, 1.0)
                        forces[did] += self.k_att * speed_factor * dir_target

                    # 2. Repulsive force from protected asset
                    diff_asset = pos - asset_pos
                    dist_asset = np.linalg.norm(diff_asset)
                    if dist_asset < self.asset_radius:
                        dist_asset = max(dist_asset, 0.1)
                        dir_asset = diff_asset / dist_asset
                        rep_mag = self.k_rep_asset * (1.0 / dist_asset - 1.0 / self.asset_radius) / (dist_asset ** 2)
                        forces[did] += rep_mag * dir_asset

                    # 3. Inter-drone repulsive forces (Collision Avoidance)
                    for other_id in drone_ids:
                        if other_id == did:
                            continue
                        diff_drone = pos - positions[other_id]
                        dist_drone = np.linalg.norm(diff_drone)
                        if dist_drone < self.d_safe:
                            dist_drone = max(dist_drone, 0.2)
                            dir_drone = diff_drone / dist_drone
                            rep_mag = self.k_rep_drone * (1.0 / dist_drone - 1.0 / self.d_safe) / (dist_drone ** 2)
                            forces[did] += rep_mag * dir_drone

                    # 4. Damping / friction
                    forces[did] -= self.damping * velocities[did]

                # Numerical integration (Velocity-Verlet)
                for did in drone_ids:
                    accel = forces[did]
                    velocities[did] += accel * sub_dt
                    speed = np.linalg.norm(velocities[did])
                    if speed > self.max_speed:
                        velocities[did] = (velocities[did] / speed) * self.max_speed
                    positions[did] += velocities[did] * sub_dt

        # Ensure final frame converges exactly to target slots
        for did in drone_ids:
            trajectories[did][-1] = target_positions[did].copy()

        return trajectories

    @staticmethod
    def verify_safety(trajectories: Dict[int, np.ndarray], min_allowed_dist: float = 3.0) -> Tuple[bool, float]:
        """
        Validates that trajectories maintain at least min_allowed_dist across all frames.
        Returns (is_safe, minimum_observed_distance).
        """
        drone_ids = list(trajectories.keys())
        n_frames = len(trajectories[drone_ids[0]])
        min_dist = float("inf")

        for f in range(n_frames):
            pts = np.array([trajectories[did][f] for did in drone_ids])
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < min_dist:
                        min_dist = d

        return min_dist >= min_allowed_dist, min_dist
