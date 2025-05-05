from environments.hexapod_env import HexapodEnv
from stable_baselines3 import PPO
import socket
import sys
import os


def close(self):
    if hasattr(self, 'sock'):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except:
            pass
        self.sock.close()
    if hasattr(self, 'webots_process'):
        self.webots_process.terminate()
        self.webots_process.wait()



env = HexapodEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("hexapod_ppo_model")
env.close()
