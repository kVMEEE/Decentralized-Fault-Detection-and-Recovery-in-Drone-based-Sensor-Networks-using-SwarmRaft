import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any

class ActiveGDOPDefenseEnv(gym.Env):
    """
    SwarmRaft Reinforcement Learning Environment:
    Coordinates honest drones to minimize 3D GDOP around a protected asset
    while strictly avoiding collisions and minimizing repositioning flight costs.
    """
    def __init__(self, n_drones: int = 7, max_speed: float = 2.0, d_safe: float = 4.0, n: Any = None):
        super(ActiveGDOPDefenseEnv, self).__init__()
        self.n_drones = n if n is not None else n_drones
        self.max_speed = max_speed
        self.d_safe = d_safe

        # Observation Space:
        # - Positions of (n_drones) surviving drones: shape (n_drones, 3) -> 3*n_drones
        # - Protected asset position: shape (3,)
        # - Current 3D GDOP score: shape (1,)
        obs_dim = (self.n_drones * 3) + 3 + 1
        self.observation_space = spaces.Box(
            low=-200.0, high=200.0, shape=(obs_dim,), dtype=np.float32
        )

        # Action Space:
        # 3D velocity adjustment vectors for each surviving drone: shape (n_drones, 3)
        action_dim = self.n_drones * 3
        self.action_space = spaces.Box(
            low=-self.max_speed, high=self.max_speed, shape=(action_dim,), dtype=np.float32
        )

        self.asset_pos = np.array([50.0, 50.0, 15.0], dtype=np.float32)
        self.drone_positions = None
        self.current_step = 0
        self.max_steps = 50

    def _calc_gdop(self, positions: np.ndarray, target: np.ndarray) -> float:
        H = []
        for p in positions:
            diff = target - p
            d = max(float(np.linalg.norm(diff)), 1e-4)
            H.append([diff[0] / d, diff[1] / d, diff[2] / d, 1.0])
        if len(H) < 4:
            return 99.9
        Hm = np.array(H)
        try:
            Q = np.linalg.inv(Hm.T @ Hm)
            trace = np.trace(Q)
            return float(min(np.sqrt(trace), 99.9)) if trace > 0 else 99.9
        except np.linalg.LinAlgError:
            return 99.9

    def _get_obs(self, gdop: float) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        obs[:self.n_drones * 3] = self.drone_positions.flatten()
        obs[self.n_drones * 3: self.n_drones * 3 + 3] = self.asset_pos
        obs[-1] = np.clip(gdop, 0.0, 50.0)
        return obs

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0

        # Initial geometric perimeter with random perturbation
        angles = np.linspace(0, 2 * np.pi, self.n_drones, endpoint=False)
        radius = 30.0
        z_base = 25.0
        z_span = 8.0

        pts = []
        for i, a in enumerate(angles):
            x = self.asset_pos[0] + radius * np.cos(a) + np.random.uniform(-2.0, 2.0)
            y = self.asset_pos[1] + radius * np.sin(a) + np.random.uniform(-2.0, 2.0)
            z = z_base + z_span * np.sin(a * 2) + np.random.uniform(-1.0, 1.0)
            pts.append([x, y, z])

        self.drone_positions = np.array(pts, dtype=np.float32)
        initial_gdop = self._calc_gdop(self.drone_positions, self.asset_pos)
        return self._get_obs(initial_gdop), {"gdop": initial_gdop}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        action_reshaped = action.reshape((self.n_drones, 3))

        # Apply displacement
        self.drone_positions += action_reshaped

        # Keep within reasonable envelope around asset
        for i in range(self.n_drones):
            d_to_asset = np.linalg.norm(self.drone_positions[i, :2] - self.asset_pos[:2])
            if d_to_asset > 50.0:
                self.drone_positions[i, :2] = self.asset_pos[:2] + (self.drone_positions[i, :2] - self.asset_pos[:2]) * (50.0 / d_to_asset)
            self.drone_positions[i, 2] = np.clip(self.drone_positions[i, 2], 12.0, 45.0)

        # 1. GDOP calculation
        new_gdop = self._calc_gdop(self.drone_positions, self.asset_pos)

        # 2. Reward formulation:
        # Base: lower GDOP is better
        reward = -(new_gdop - 2.0) * 5.0

        # Repositioning effort penalty
        effort = np.sum(action_reshaped ** 2) * 0.1
        reward -= effort

        # Collision penalty
        collision_penalty = 0.0
        for i in range(self.n_drones):
            for j in range(i + 1, self.n_drones):
                d = np.linalg.norm(self.drone_positions[i] - self.drone_positions[j])
                if d < self.d_safe:
                    collision_penalty += 30.0 * (self.d_safe - d)
        reward -= collision_penalty

        # Termination conditions
        terminated = False
        if new_gdop < 2.2 and collision_penalty == 0.0:
            reward += 50.0
            terminated = True

        truncated = self.current_step >= self.max_steps
        return self._get_obs(new_gdop), reward, terminated, truncated, {"gdop": new_gdop, "collisions": collision_penalty > 0}
