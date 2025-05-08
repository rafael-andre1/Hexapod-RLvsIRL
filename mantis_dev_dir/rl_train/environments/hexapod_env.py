# environments/hexapod_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
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
    def __init__(self, task='walk'):
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
        self.webots_process = subprocess.Popen([
            r"C:\Users\hasht\AppData\Local\Programs\Webots\msys64\mingw64\bin\webots.exe",
            "--stdout",
            # "--no-rendering",
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

    def get_initial_observation(self):
        # TODO: Currently not used, should be straightforward to define
        # as the starting point should be easy to generate
        return np.zeros(29)  
    
    def check_done(self, com, step_count, max_steps=500):
        # TODO: optimize and add conditions:
        # - time step limit is reached
        # - body too low

        # TODO: should these conditions stop episode when its really bad
        # or when it has achieved what we are looking for? both?
        # If good -> nothing else to learn
        # If terrible for a long time -> fresh start
        if (step_count >= max_steps) or (com is not None and com[2] < 0.015):
            return True
        return False


    def compute_rewards(self, obs):
        imu_data = obs['imu']  # [theta, acc]
        com = obs['com']       # [x, y, z]
        foot_contacts = obs['foot_contacts']  # [foot1, foot2, ... , foot6]

        theta, acc = imu_data[0], imu_data[1]
        com_height = com[2]

        # TODO: needs fine tuning and actual sensor values 
        # Reward depends on task
        if self.task == 'stand_up':
            h_base = 1.0  # assuming h=1 as acceptable height
            vcom = abs(com_height - h_base) / h_base
            reward = 1.0 - vcom
            # done = vcom > 0.3

        elif self.task == 'walk':
            dx = com[0] - self.prev_com[0] if self.prev_com else 0
            stability = abs(theta) + abs(acc)
            reward = dx - 0.1 * stability
            # done = stability > 5.0

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

        # TODO: how do I only reset position every episode instead of iteration
        # efficiently? Currently lazy and maybe incorrect implementation in the controller.
        self.conn.sendall(json.dumps({'command': 'reset'}).encode('utf-8'))

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