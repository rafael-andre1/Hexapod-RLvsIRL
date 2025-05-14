from environments.hexapod_env import HexapodEnv
from stable_baselines3 import PPO

from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback

import torch

# Custom callbacks (times function for runtime prediction)
class TqdmCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.progress_bar = None

    def _on_training_start(self):
        self.progress_bar = tqdm(total=self.locals['total_timesteps'])

    def _on_step(self):
        self.progress_bar.update(1)
        return True

    def _on_training_end(self):
        self.progress_bar.close()
        self.progress_bar = None


# -------------------------- Training -------------------------- #

# Environment setup
env = HexapodEnv()

# Select gpu if available, otherwise cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("############################")
print("#######   WARNING    #######")
print("############################")
print("ACTIONS CURRENTLY DISABLE FOR INITIAL OBS READING")


# Model choice
model = PPO("MlpPolicy", env, verbose=1, device=device)

# Model training
model.learn(total_timesteps=10000, callback=TqdmCallback())
model.save("hexapod_ppo_model")
env.close()
