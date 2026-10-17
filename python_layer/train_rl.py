import os
import sys
import numpy as np
from stable_baselines3 import PPO

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from rl_defense_env import ActiveGDOPDefenseEnv

def train_and_save_policy(total_timesteps: int = 15000):
    models_dir = os.path.join(_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "gdop_defense_policy.zip")

    print(f"[SwarmRaft RL] Initializing 3D Active GDOP Defense Environment (n=7)...")
    env = ActiveGDOPDefenseEnv(n_drones=7, max_speed=1.5, d_safe=4.0)

    print(f"[SwarmRaft RL] Training PPO Policy over {total_timesteps} timesteps...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        seed=42
    )

    model.learn(total_timesteps=total_timesteps)

    model.save(model_path)
    print(f"[SwarmRaft RL] Model successfully saved to: {model_path}")

    # Evaluate trained policy
    obs, info = env.reset(seed=123)
    init_gdop = info["gdop"]
    for _ in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
        if terminated:
            break

    final_gdop = step_info["gdop"]
    print(f"[SwarmRaft RL] Evaluation complete: Initial GDOP={init_gdop:.2f} -> Optimized GDOP={final_gdop:.2f}")

if __name__ == "__main__":
    train_and_save_policy(total_timesteps=15000)
