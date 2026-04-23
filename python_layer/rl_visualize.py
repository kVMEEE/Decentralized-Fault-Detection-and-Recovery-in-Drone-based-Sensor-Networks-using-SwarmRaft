import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from rl_defense_env import ActiveGDOPDefenseEnv

def visualize_defense_2d():
    print("Loading RL Environment...")
    n_drones = 15 
    env = ActiveGDOPDefenseEnv(n=n_drones, max_speed=5.0)
    
    print("Training Model...")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=15000)
    
    # Custom initial state: Terribly clustered on one side
    target = np.array([0.0, 0.0, 0.0]) # The compromised drone is at Origin
    honest_int = np.array([[20.0 + (i*0.5), 15.0 - (i*0.5), 0.0] for i in range(n_drones-1)], dtype=np.float32)
    
    obs = np.zeros(env.observation_space.shape, dtype=np.float32)
    obs[:(n_drones-1)*3] = honest_int.flatten()
    obs[(n_drones-1)*3 : (n_drones-1)*3 + 3] = target
    initial_gdop = env._calculate_gdop(honest_int, target)
    obs[-1] = initial_gdop
    env.state = obs
    
    print("Executing RL Model Active Defense Reconfiguration...")
    steps = 0
    while steps < 100:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        
    honest_final = obs[:(n_drones-1)*3].reshape((n_drones-1, 3))
    final_gdop = obs[-1]
    
    # Produce the exact Graph style from main.py Phase 1!
    plt.figure(figsize=(12, 8))
    
    # Extract 2D Projections (X and Y coordinates)
    target_x = target[0]
    target_y = target[1]
    
    start_x = honest_int[:, 0]
    start_y = honest_int[:, 1]
    
    final_x = honest_final[:, 0]
    final_y = honest_final[:, 1]

    # Plot Target (The 'True' anchor we are trying to bracket)
    plt.scatter([target_x], [target_y], marker='x', color='green', s=150, label='Compromised Target')
    
    # Plot Pre-Defense (Mapping to the old "Spoofed/Noisy" vibe)
    plt.scatter(start_x, start_y, marker='^', color='red', s=80, label='Initial Cluster (Bad GDOP)')
    
    # Plot Post-Defense (RL Recovered Form)
    plt.scatter(final_x, final_y, marker='*', color='gold', s=150, edgecolors='black', label='RL Optimized Formation')

    # Draw arrows from the initial bad positions to the RL optimized envelope
    for i in range(n_drones - 1):
        plt.arrow(start_x[i], start_y[i], final_x[i] - start_x[i], final_y[i] - start_y[i], 
                  color='gray', linestyle=':', head_width=0.8, alpha=0.5)

    plt.title(f"SwarmRaft RL Defense Result (n={n_drones}, Initial GDOP={initial_gdop:.1f} -> Final GDOP={final_gdop:.1f})")
    plt.xlabel("Projected X (m)")
    plt.ylabel("Projected Y (m)")
    plt.legend()
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    visualize_defense_2d()
