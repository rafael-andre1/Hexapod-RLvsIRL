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


def is_port_in_use(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0
    
# Useful for debugging
if is_port_in_use(5000):
    raise RuntimeError("Port 5000 is already in use. Please close existing process.")


class HexapodEnv(gym.Env):
    def __init__(self, task='stand_up'):
        super().__init__()
        self.task = task
        # Action space -> 18 actuators
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(18,), dtype=np.float32)

        # Observation Space -> 29 values

        """
          - Actuator position readings: 18
 
          - IMU (angle + acceleration): 2
 
          - Foot contacts: 6
 
          - Center of mass (3D vector): 3
        
        """
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)

        # Start Webots simulation
        # webots_cmd = os.environ.get("WEBOTS_CMD", "webots")
        webots_cmd = r"C:\Users\hasht\AppData\Local\Programs\Webots\msys64\mingw64\bin\webots.exe"

        self.webots_process = subprocess.Popen([
            webots_cmd,
            "--stdout",
            "worlds/mantis.wbt"
        ])
        time.sleep(5)

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
        self.prev_com = None
        self.prev_pos = 0
        self.total_steps = 0
        self.stable_counter = 0

    def get_initial_observation(self):
        # TODO: Currently not used, should be straightforward to define
        # as the starting point should be easy to generate
        return np.zeros(29)  
    
    def check_done(self, com, step_count, max_steps=500):
        # TODO: optimize and add conditions:
        # - time step limit is reached
        # - body too low

        # TODO: should these conditions stop episode when its really bad or...
        # TODO: when it has achieved what we are looking for? both?
        # If good -> nothing else to learn
        # If terrible for a long time -> fresh start
        if (step_count >= max_steps):
            return True
        return False


    def compute_rewards(self, obs):
        imu_data = obs['imu']  # [theta, acc]
        com = obs['com']       # [x, y, z]
        foot_contacts = obs['foot_contacts']  # [foot1, foot2, ... , foot6]
        lidar_values_original = obs['lidar']
        joint_sensors = obs['joint_sensors']

        # Sanitizing values to avoid inf when robot flips over
        lidar_values = [v for v in lidar_values_original if math.isfinite(v)]

        theta, acc = imu_data[0], imu_data[1]
        com_height = com[2]

        # TODO: needs fine tuning and actual motor position sensor values
        if self.task == 'stand_up':
            reward = 0 # starts at zero, based on conditions changes value

            # Acceptable height + stability at height
            h_base = 4.5 # empirically defined as reasonable height
            diff = abs(max(lidar_values) - h_base)
            if diff <= 0.4:
                self.stable_counter += 1
                # The more stable, the higher the reward
                # In order to avoid explosive increase,
                # we consider 20% of total steps being stable
                # as multiplier for reward

                reward += 1 * (0.2 * self.stable_counter)
            else:
                reward -= 0.5
                self.stable_counter = 0

            # For every foot that's not touching the ground, we take points
            for v in foot_contacts:
                if v == 0: reward -= 0.5
                elif v == 1: reward += 1
                else: print("NON-READABLE FOOT SENSOR VALUE! ", v)

            """
             Following the mantis tutorial, after reading the .wbt
              file values for the hinge position:
               - if the "elbow" hinges were to be perfectly bent/balanced, 
               its angle would be [ ~ -2.4121293759260714 rad -> ~ -138.2 deg ]
            
             Therefore, the lower this negative number is, the tighter the 
              robot closes its arm.
            
             In order to, again, respect a threshold as it was done in
              the height check.
            """

            # Acceptable arm position (hinge safety)
            base_angle = -138.2
            for joint_angle in joint_sensors:
                diff = abs(joint_angle - base_angle)
                if diff <= 10:
                    # In order to enforce stability while
                    # standing, this reward is much more important (3x)
                    reward += 1 * 3
                else:
                    reward -= 0.5 * 3


        elif self.task == 'walk':
            # === BASE HEIGHT ===
            h_base = 1.0  # expected standing height
            h_error = abs(com_height - h_base)

            # Reward for being upright (standing height)
            reward_height = max(0.0, 1.0 - h_error / 0.2)  # normalized (0 to 1), tolerant to ±0.2m

            # === STABILITY ===
            max_theta = 0.5  # radians (≈28°)
            reward_stability = max(0.0, 1.0 - abs(theta) / max_theta)

            # === CONTACT POINTS ===
            feet_on_ground = sum(1 for contact in foot_contacts if contact > 0.5)  # threshold to avoid noise
            reward_feet = feet_on_ground / 6.0  # encourage all feet on ground

            # === FINAL REWARD ===
            reward = (
                    0.5 * reward_height +  # prioritize height
                    0.3 * reward_stability +  # stability also matters
                    0.2 * reward_feet  # contact is important but less critical
            )

        elif self.task == 'climb':
            delta_step = 1 if com[2] > self.prev_com[2] + 0.05 else 0
            reward = delta_step
            # done = delta_step == 0 and self.total_steps > 10

        else:
            reward = 0
            # done = False

        self.prev_com = com
        return reward

    def step(self, action):
        # Sends actions into Webots
        print("Action sent by PPO: ", action)
        self.conn.sendall(json.dumps(action.tolist()).encode('utf-8'))

        # Pulls readings after actions
        data = self.conn.recv(4096)
        obs = json.loads(data.decode('utf-8'))

        # Transforms into numpy array for efficiency of reward calculations
        observation = np.array(
            obs['joint_sensors'] + obs['imu'] + obs['foot_contacts'] + obs['com'],
            dtype=np.float32
        )

        """
        print("Joint sensors:", obs['joint_sensors'])
        print("IMU:", obs['imu'])
        print("Foot contacts:", obs['foot_contacts'])
        print("Center of mass:", obs['com'])
        """

        com = obs['com']

        reward = self.compute_rewards(obs)
        
        # TODO: Currently assumes 20 episodes of 500 steps each
        # (10.000 timesteps / 500 max steps -> 20 episodes)
        done = self.check_done(com, self.total_steps)
        self.total_steps += 1

        return observation, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.total_steps = 0
        self.prev_com = None

        # TODO: how do I only reset position every episode instead of iteration? Solved?
        # efficiently? Currently lazy and maybe incorrect implementation in the controller.
        self.conn.sendall((json.dumps({'command': 'reset'}) + "\n").encode('utf-8'))

        # Wait for "reset_complete" confirmation
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

        # TODO: how do I pull original positions for every reset?
        initial_obs = np.zeros(29, dtype=np.float32)
        return initial_obs, {}
    

    def close(self):
        # Shuts down connection and terminates socket
        if hasattr(self, 'conn'):
            self.conn.close()
        if hasattr(self, 'sock'):
            self.sock.close()


"""
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
"""