from environments.hexapod_env import HexapodEnv
from stable_baselines3 import PPO
import sys
import os

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

env = HexapodEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("hexapod_ppo_model")
env.close()
