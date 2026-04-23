import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ActiveGDOPDefenseEnv(gym.Env):
    """ SwarmRaft RL Env: Reposition honest drones to minimize GDOP for multilateration. """
    def __init__(self, n=5, max_speed=2.0):
        super(ActiveGDOPDefenseEnv, self).__init__()
        
        self.n = n 
        self.max_velocity = max_speed
        
        # State Space: Positions of (n-1) honest drones, 1 estimated target, 1 GDOP score
        obs_dim = (self.n - 1) * 3 + 3 + 1
        self.observation_space = spaces.Box(low=-1000.0, high=1000.0, shape=(obs_dim,), dtype=np.float32)
        
        # Action Space: Velocity vectors for the (n-1) honest drones
        action_dim = (self.n - 1) * 3
        self.action_space = spaces.Box(low=-self.max_velocity, high=self.max_velocity, shape=(action_dim,), dtype=np.float32)
        
        self.state = None
        
    def _calculate_gdop(self, honest_positions, target_position):
        """ Mathematical GDOP penalty calculation. """
        H = []
        for pos in honest_positions:
            dist = np.linalg.norm(pos - target_position)
            if dist < 1e-6: dist = 1e-6 
            
            unit_vec = (target_position - pos) / dist
            H.append([unit_vec[0], unit_vec[1], unit_vec[2], 1.0]) 
            
        H = np.array(H)
        try:
            # Sqrt(Trace((H^T * H)^-1))
            gdop = np.sqrt(np.trace(np.linalg.inv(H.T @ H)))
        except np.linalg.LinAlgError:
            gdop = 200.0 
            
        return gdop

    def step(self, action):
        action = action.reshape((self.n - 1, 3))
        
        honest_positions = self.state[:(self.n - 1)*3].reshape((self.n - 1, 3))
        target_position = self.state[(self.n - 1)*3 : (self.n - 1)*3 + 3]
        
        # Agentic Action: Shift geographic formation
        new_honest_positions = honest_positions + action
        
        # Calculate new GDOP penalty
        new_gdop = self._calculate_gdop(new_honest_positions, target_position)
        
        # Reward Function: Penalize high GDOP to enforce geometric optimization
        reward = -new_gdop 
        
        terminated = False
        if new_gdop < 1.5: # Excellent GDOP Achieved
            reward += 200.0
            terminated = True
            
        # Update State Space
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.state[:(self.n - 1)*3] = new_honest_positions.flatten()
        self.state[(self.n - 1)*3 : (self.n - 1)*3 + 3] = target_position
        self.state[-1] = new_gdop
            
        return self.state, reward, terminated, False, {"gdop": new_gdop}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        target_pos = np.random.uniform(-5, 5, size=3)
        honest_pos = np.random.uniform(-40, 40, size=(self.n - 1, 3))
        
        initial_gdop = self._calculate_gdop(honest_pos, target_pos)
        
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.state[:(self.n - 1)*3] = honest_pos.flatten()
        self.state[(self.n - 1)*3 : (self.n - 1)*3 + 3] = target_pos
        self.state[-1] = initial_gdop
        
        return self.state, {}

if __name__ == "__main__":
    from stable_baselines3 import PPO
    print("Initializing RL Environment for SwarmRaft Active Defense Training...")
    env = ActiveGDOPDefenseEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    
    print("Training the Swarm to recognize and actively reconfigure geometries for GDOP Optimization...")
    model.learn(total_timesteps=3000)
    print("RL Model Training Complete. Ready for network deployment.")
