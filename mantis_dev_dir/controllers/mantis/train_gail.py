#!/usr/bin/env python3
# train_gail.py

import os
import sys

# ────────────────────────────────────────────────────────────────
# Prepend the ‘mantis_dev_dir’ directory so we can import rl_train.environments
# train_gail.py lives in mantis_dev_dir/controllers/mantis/
# we want mantis_dev_dir on the path, two levels up:
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
sys.path.insert(0, project_root)
# ────────────────────────────────────────────────────────────────

import argparse

import gymnasium as gym
from gymnasium.error import NameNotFound
from gymnasium.envs.registration import register, spec

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.data.types import TrajectoryWithRew
from imitation.util.logger import configure

def load_expert(csv_path, motor_cols, obs_cols):
    df = pd.read_csv(csv_path)
    acts = df[motor_cols].to_numpy()
    obs = df[obs_cols].to_numpy()
    obs_padded = np.vstack([obs, obs[-1:]])
    rews = np.zeros(len(acts), dtype=float)
    infos = [{} for _ in range(len(acts))]
    return TrajectoryWithRew(
        obs=obs_padded,
        acts=acts,
        infos=infos,
        terminal=True,
        rews=rews,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",         default="expert_data.csv",
                        help="Path to your expert CSV log")
    parser.add_argument("--env-id",      default="HexapodEnv-v0",
                        help="Gym ID for your custom env")
    parser.add_argument("--entry-point", default="rl_train.environments.hexapod_env:HexapodEnv",
                        help="module:Class for your env")
    parser.add_argument("--max-steps",   type=int, default=500,
                        help="max_episode_steps for the env")
    parser.add_argument("--log-dir",     default="./gail_logs",
                        help="Where to save logs and models")
    parser.add_argument("--timesteps",   type=int, default=200_000,
                        help="Total timesteps for GAIL training")
    parser.add_argument("--demo-batch",  type=int, default=128,
                        help="Batch size for sampling expert demos")
    parser.add_argument("--disc-updates",type=int, default=4,
                        help="Discriminator updates per round")
    parser.add_argument("--disc-batch",  type=int, default=128,
                        help="Discriminator batch size")
    args = parser.parse_args()

    # 1) Register the custom env if missing
    try:
        spec(args.env_id)
    except NameNotFound:
        print(f"[Info] Registering {args.env_id} → {args.entry_point}")
        register(
            id=args.env_id,
            entry_point=args.entry_point,
            max_episode_steps=args.max_steps,
        )

    # 2) Logging setup
    os.makedirs(args.log_dir, exist_ok=True)
    configure(folder=args.log_dir, format_strs=["stdout", "tensorboard"])

    # 3) Split CSV columns into actions vs. observations
    MOTOR_NAMES = [
        "RPC","RPF","RPT","RMC","RMF","RMT",
        "RAC","RAF","RAT","LPC","LPF","LPT",
        "LMC","LMF","LMT","LAC","LAF","LAT"
    ]
    df0 = pd.read_csv(args.csv, nrows=1)
    all_cols = df0.columns.tolist()
    obs_cols = [c for c in all_cols if c not in ["time"] + MOTOR_NAMES]

    # 4) Load the expert trajectory
    demo_traj = load_expert(args.csv, MOTOR_NAMES, obs_cols)

    # 5) Create the Gym environment
    env = gym.make(args.env_id)

    # 6) Set up PPO as the generator
    gen_algo = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=os.path.join(args.log_dir, "ppo_tb"),
    )

    # 7) Build the discriminator/reward network
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )

    # 8) Instantiate and train GAIL
    gail = GAIL(
        demonstrations=[demo_traj],
        demo_batch_size=args.demo_batch,
        gen_algo=gen_algo,
        n_disc_updates_per_round=args.disc_updates,
        disc_batch_size=args.disc_batch,
        reward_net=reward_net,
        log_dir=args.log_dir,
        rng=None,
    )
    gail.train(total_timesteps=args.timesteps)

    # 9) Save the trained policy
    gen_algo.save(os.path.join(args.log_dir, "gail_hexapod_policy.zip"))
    print(f"Training complete. Logs & model saved in {args.log_dir}")

if __name__ == "__main__":
    main()
