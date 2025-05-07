from environments.hexapod_env import HexapodEnv
from stable_baselines3 import PPO
import socket
import sys
import os


env = HexapodEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("hexapod_ppo_model")
env.close()
