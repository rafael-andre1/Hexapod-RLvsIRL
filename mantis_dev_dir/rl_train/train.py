from environments.hexapod_env import HexapodEnv
from stable_baselines3 import PPO

from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback

import pandas as pd
import torch
import os

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

# Task Choice

task = input("What's the task? ")

# Environment setup
env = HexapodEnv(task)

# Select gpu if available, otherwise cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print("\n#########################################")
print(f"#######  Using device: {device}   #######")
print("#########################################\n")


"""
print("############################")
print("#######   WARNING    #######")
print("############################")
print("ACTIONS CURRENTLY DISABLED FOR INITIAL OBS READING")
"""

# ------------------------- Model Choice ---------------------- #



expert_choice = input("Would you like to use an expert?")


# Raw RL
if expert_choice != "yes":

    print("\n-----------------------------")
    print("--- NOT Using Expert !!! ----")
    print("-----------------------------\n")

    if task == "walk":
        choice = str(input("Would you like to use transfer learning for walking? "))
        model_path=r"C:\Users\hasht\Desktop\saved_model"
        if choice == "yes": model = PPO.load(model_path+"\\hexapod_ppo_bestStandUp", env=env, device=device)
    else: model = PPO("MlpPolicy", env, verbose=1, device=device)

    # Model training
    if task == "stand_up": model.learn(total_timesteps=180000, callback=TqdmCallback())
    else: model.learn(total_timesteps=250000, callback=TqdmCallback()) # for walk
    model.save("hexapod_ppo_model")
    env.close()

# Using Expert
elif task == "walk":
    # Expert Setup

    from imitation.data.types import Trajectory
    df = pd.read_csv(r"gail_data\expert_data.csv")

    # Separate obs and values
    acts = df.iloc[:, 1:19].values # Starts at index 1, finishes at index 19-1 == 18
    obs = df.iloc[:, 19:].values # All remaining values are observations
    time = df.iloc[:, 0].values # Time is the first column

    # Dropping the last action is very important,
    # as the discriminator needs to know what obs
    # the action will generate. This guarantees that.
    acts = acts[:-1]

    # Now that we have the actions, we can define what's called a trajectory
    traj = Trajectory(obs=obs, acts=acts, infos=None, terminal=True)
    traj_list = [traj]

    from imitation.algorithms.adversarial.gail import GAIL
    from imitation.rewards.reward_nets import BasicRewardNet
    from stable_baselines3.common.vec_env import DummyVecEnv

    print("\n-----------------------------")
    print("------- Using Expert !!! ----")
    print("-----------------------------\n")



    # Base model
    policy_kwargs = dict(net_arch=[64, 64])

    # Learner model
    learner = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)

    # Computes GAIL-based rewards
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    # GAIL is the discriminator. It automatically blocks the learner from getting rewards, and
    # analyzes its behaviour, comparing it to the expert's. GAIL will influence the learner
    # to lean towards the perfect values, by giving it meaningful rewards.
    gail_trainer = GAIL(
        demonstrations=traj_list,
        venv=DummyVecEnv([lambda:env]), # Very important!
        gen_algo=learner,
        demo_batch_size=64,  # Stable Choice
        reward_net=reward_net
    )

    # Currently testing only
    gail_trainer.train(75000)
    gail_trainer.gen_algo.save("gail_hexapod_model")

else: print("Not possible to do that task while using an expert. Sorry!")