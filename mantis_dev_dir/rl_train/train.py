# ---------------- Requirements ---------------- #

# RL
from environments.hexapod_env import HexapodEnv
from stable_baselines3 import PPO, A2C, DDPG

# IRL
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.types import Trajectory
import pandas as pd

# Benchmarking
import os
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback

# Device
import torch
# ---------------------------------------------- #


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

def get_unique_model_name(base_name):
    counter = 1
    while os.path.exists(f"{base_name}_{counter}.zip"):
        counter += 1
    return f"{base_name}_{counter}"

# ------------------- Device Setup ------------------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
print("\n#########################################")
print(f"#######  Using device: {device}   #######")
print("#########################################\n")
# ---------------------------------------------------- #




                                # ------------------------- Model Choice ---------------------- #

# Task Choice
task = input("What's the task? ")

# Model Choice
model_choice = input("Which model do you want to use? (PPO, A2C, DDPG) ")

# Transfer Learning Option
choice = str(input("Would you like to use transfer learning for walking? (No DDPG!) "))

# Expert Option
expert_choice = input("Would you like to use an expert? (Only works with PPO and A2C!) ")
expert_choice = True if expert_choice == "yes" else False


# Environment setup
env = HexapodEnv(task, model_choice, expert_choice)

# Raw RL
if not expert_choice:

    print("\n-----------------------------")
    print("--- NOT Using Expert !!! ----")
    print("-----------------------------\n")

    if task == "walk":
        model_path=r"C:\\Users\\hasht\\Desktop\\stand_up - models and logs"
        if choice == "yes":
            model = PPO.load(model_path+f"\\hexapod_{model_choice}_model_1", env=env, device=device, verbose=0)

        try:
            model.learn(total_timesteps=250000, callback=TqdmCallback())
        finally:
            # Always runs, even on interrupt or crash 
            print("Saving model and closing environment...")
            
            # Avoid overwrite
            base_model_name = f"hexapod_{model_choice}_model"
            unique_model_name = get_unique_model_name(base_model_name)
            model.save(unique_model_name)
            env.close()


    elif task == "stand_up":
        if model_choice == "PPO": model = PPO("MlpPolicy", env, verbose=1, device=device)
        elif model_choice == "A2C": model = A2C("MlpPolicy", env, verbose=1, device=device)
        elif model_choice == "DDPG": model = DDPG("MlpPolicy", env, verbose=1, device=device)
        else: print("No model with such designation!")

        try:
            model.learn(total_timesteps=180000, callback=TqdmCallback())
        finally:
            # Always runs, even on interrupt or crash 
            print("Saving model and closing environment...")
            
            # Avoid overwrite
            base_model_name = f"hexapod_{model_choice}_model"
            unique_model_name = get_unique_model_name(base_model_name)
            model.save(unique_model_name)
            env.close()

# Inverse RL
elif task == "walk":
    
    # Expert Setup
    
    df = pd.read_csv(r"gail_data\clean_expert.csv")

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

    print("\n-----------------------------")
    print("------- Using Expert !!! ----")
    print("-----------------------------\n")



    # Base model
    policy_kwargs = dict(net_arch=[64, 64])

    # Learner model
    model_path = r"saved_models\stand_up - models and logs"
    if choice == "yes":
        if model_choice == "PPO":
            learner = PPO.load(model_path + f"\\hexapod_{model_choice}_model_1", env=env, device=device, verbose=0)
        else:
            learner = A2C.load(model_path + f"\\hexapod_{model_choice}_model_1", env=env, device=device, verbose=0)
    else: learner = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)

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
    gail_trainer.train(150000)
    gail_trainer.gen_algo.save("gail_hexapod_model")

else: print("Not possible to do that task while using an expert. Sorry!")