# environments/hexapod_env.py
import gymnasium as gym
from gymnasium import spaces
import os
import numpy as np
import random
import socket
import math
import json
import subprocess
import time
import csv
from pathlib import Path
import inspect



# 6, observation space should not include the 18 actions
OBS_SPACE_SIZE = 6


def is_port_in_use(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0
    
# Useful for debugging
if is_port_in_use(5000):
    raise RuntimeError("Port 5000 is already in use. Please close existing process.")


class HexapodEnv(gym.Env):
    def __init__(self, task, model="PPO", expert=False):
        super().__init__()
        self.task = task
        self.expert = expert
        self.model = model
        if self.expert: print("Environment recognizes expert!")

        # Action space -> 18 actuators
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(18,), dtype=np.float32)

        # Observation Space -> 6 values
        self.observation_space = spaces.Box(low=-1, high=1, shape=(OBS_SPACE_SIZE,), dtype=np.float32)
        """ 
          - IMU (roll, pitch, yaw): 3
          - Robot Translation (x, y, z): 3        
        """


        # Start Webots simulation
        webots_cmd = r"C:\Users\hasht\AppData\Local\Programs\Webots\msys64\mingw64\bin\webots.exe"
        self.webots_process = subprocess.Popen([
            webots_cmd,
            "--stdout",
            "worlds/mantis.wbt"
        ])
        time.sleep(2)

        # Set up socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(('127.0.0.1', 5000))
        self.sock.listen(1)

        # Awaits controller link attempt
        while True:
            try:
                print("Awaiting controller socket link...")
                self.conn, _ = self.sock.accept()
                break
            except socket.error:
                time.sleep(0.1)

        # Reward function metrics
        self.is_tilted = True
        self.total_steps = 0
        self.cur_overall_step = 0
        self.cur_dist = 0

        # Gail
        self.num_envs = 1

        # Writing to CSV
        self.csv_writer = None
        self.csv_file = None

        # Setup logging file (no overwrites)
        filename_base = f"{self.task}_{self.model}_{'IRL' if self.expert else 'RL'}"
        file_dir = Path("logs")
        file_dir.mkdir(exist_ok=True)
        suffix = 0
        while True:
            file_path = file_dir / f"{filename_base}_{suffix}.csv"
            if not file_path.exists():
                break
            suffix += 1

        self.csv_file = open(file_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "cur_step", "mantis_x", "mantis_y", "mantis_z",
            "stability_reward", "height_reward", "walk_reward", "total_reward"
        ])

    def check_done(self, step_count, max_steps=600):
        if step_count % 100 == 0: print("Step: ", step_count)

        # "Done" is only for step limit, otherwise -> reward leak
        if (self.task == "stand_up" and (step_count > max_steps)): return True
        if (self.task == "walk" and (step_count > max_steps*3)): return True
        return False

    def _log_step(self, x, y, z, stab, height, walk, total):
        if self.csv_writer:
            self.csv_writer.writerow([
                self.cur_overall_step, x, y, z, stab, height, walk, total
            ])

    
    def checkTilt(self, pitch, roll):
        # Tilt control helps avoid flipping over
        # 0.1 radians ~ 5.7 degrees
        self.is_tilted = True
        
        if abs(roll) < 0.1 and abs(pitch) < 0.1:
            self.is_tilted = False
            return 2  # Great stability
        
        elif abs(roll) < 0.3 and abs(pitch) < 0.3:
            return 1  # Good stability
        
        elif abs(roll) < 0.6 and abs(pitch) < 0.6:
            return -1  # Slightly unstable

        # Rolling over or being very tilted is highly penalized
        return -5
            


    def compute_rewards(self, obs):

        """ FUNCTION INFORMATION

            Rewards have (not explicitly) weights based on task importance.

            # ------- stand_up ------- #
                - stability is always preferred over height
                - stability is also more "forgiving" than height
                - penalties for high instability / bad height
                  are ~5 times the best rewards, guarantees goal

            # --------- walk --------- #
                - inherits "stand_up" rewards
                - can use transfer learning from standing up
                - can use GAIL
                - reward == twice distance walked in a straight line,
                  while remaining stable and with good height

            # --------- extra --------- #

                - For all of our ideas and experiments during development,
                please refer to "\\future_implementations\\extra_rewards.py"
        """

        # Fetching relevant values
        robot_pose = obs["robot_pose"]                     # [x, y, z]
        roll, pitch, yaw = obs['imu'][0], obs['imu'][1], obs['imu'][2]

        x, y, z = robot_pose

        stability_reward = 0
        height_reward = 0
        walk_reward = 0

        if self.task == 'stand_up' or self.task == 'walk':
            # Acceptable height + stability at height
            h_base = 3
            diff = abs(z - h_base)
            stability_reward = self.checkTilt(pitch, roll)

            # For now, Height is only rewarded if robot is stable
            if 1.9 <= diff <= 2.1 and stability_reward >= 1: height_reward = 1
            elif  2.2 <= diff < 2.6: height_reward = -0.5

            # Flipped over | Didn't stand up
            else: height_reward -= 4

            if self.task == 'walk':

                # Walking forward is -x (negative) difference and 0 y difference
                # so if I take points for negative (x plus y), I reward it
                # when it walks forward, in a straight line
                self.cur_dist = robot_pose[0]
                if stability_reward >= 1:
                    walk_reward -= 2*(robot_pose[0] + robot_pose[0])
                else: walk_reward -= 1

        # Compute total
        reward = stability_reward + height_reward + walk_reward

        # Logs every 200 time steps (easier for plotting, and still captures evolution)
        # This way, every 3 lines are equivalent to an entire episode
        if self.cur_overall_step % 200 == 0:
            self._log_step(x, y, z, stability_reward, height_reward, walk_reward, reward)

        if self.task == "walk":
            # Standing up straight is still important,
            # but only as a baseline, so we halved the reward
            reward = (stability_reward + height_reward) / 2
            reward += height_reward

        return reward

    def step(self, action):
        # Sends actions into Webots
        self.conn.sendall(json.dumps(action.tolist()).encode('utf-8'))

        # Pulls readings after actions
        data = self.conn.recv(4096)
        obs = json.loads(data.decode('utf-8'))

        # Transforms into numpy array for efficiency of reward calculations
        observation = np.array(
            obs['imu'] + obs["robot_pose"],
            dtype=np.float32
        )

        reward = self.compute_rewards(obs)
        
        # Small episode guide:
        # (10.000 timesteps / 500 max steps -> 20 episodes)
        done = self.check_done(self.total_steps)
        self.total_steps += 1
        self.cur_overall_step += 1

        return observation, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.total_steps = 0
        self.conn.sendall((json.dumps({'command': 'reset'}) + "\n").encode('utf-8'))

        # Waits for "reset_complete" confirmation
        buffer = ""
        while True:
            data = self.conn.recv(4096).decode('utf-8')
            buffer += data
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if not line.strip():
                    continue
                msg = json.loads(line)
                if isinstance(msg, dict) and msg.get("status") == "reset_complete":
                    break
            else:
                continue
            break

        initial_obs = np.zeros(OBS_SPACE_SIZE, dtype=np.float32)
        return initial_obs, {}
    

    def close(self):
        # Shuts down connection and terminates socket
        if hasattr(self, 'conn'):
            self.conn.close()
        if hasattr(self, 'sock'):
            self.sock.close()
        if self.csv_file:
            self.csv_file.close()