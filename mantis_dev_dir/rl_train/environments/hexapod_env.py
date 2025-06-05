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

OBS_SPACE_SIZE = 30


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
        self.observation_space = spaces.Box(low=-1, high=1, shape=(OBS_SPACE_SIZE,), dtype=np.float32)

        # Start Webots simulation
        # webots_cmd = os.environ.get("WEBOTS_CMD", "webots")
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
        self.prev_com = None
        self.is_tilted = True
        self.prev_pos = 0
        self.total_steps = 0
        self.stable_counter = 0
        self.starting_height = None
        self.cur_dist = 0
        
    
    def check_done(self, com, step_count, max_steps=800):
        # TODO: add other task "dones"
        # for "stand up"
        if step_count % 100 == 0:
            print("Step: ", step_count)
        if (self.task == "stand_up" and
                ((step_count >= max_steps) or
                (self.stable_counter >= 800 and self.is_tilted == False))):
            return True

        if (self.task == "walk" and
                ((step_count >= max_steps*4) or
                (self.is_tilted == False and abs(self.cur_dist) >= 10))):
            return True
        return False
    
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
        imu_data = obs['imu']                 # [theta, acc]
        com = obs['com']                      # [x, y, z]
        foot_contacts = obs['foot_contacts']  # [foot1, foot2, ... , foot6]
        robot_pose = obs["robot_pose"]        # [x, y, z]
        joint_sensors = obs['joint_sensors']
        imu_values = obs['imu']
        roll, pitch, yaw = imu_values[0], imu_values[1], imu_values[2]
        theta, acc = imu_data[0], imu_data[1]
        com_height = com[2]

        """ Lidar code
        # lidar_values_original = obs['lidar']
        # Sanitizing values to avoid inf when robot flips over
        lidar_values = [v if math.isfinite(v) else 999 for v in lidar_values_original]

        [...]

        # Acceptable height + stability at height
            h_base = 3 # empirically defined as reasonable height
            diff = abs(lidar_values[1] - h_base)
        """
        
        if self.task == 'stand_up' or self.task == 'walk':
            reward = 0 # starts at zero, based on conditions changes value

            # Acceptable height + stability at height
            h_base = 3
            diff = abs(robot_pose[2] - h_base)
            # print(diff)
            if 1.9 <= diff <= 2.1:

                # For future implementations:
                # The more stable, the higher the reward
                # In order to avoid explosive increase,
                # we consider 20% of total steps being stable
                # as multiplier for reward
                # (1 + (0.005 * self.stable_counter)) if self.is_tilted==False

                reward += 1 + self.checkTilt(pitch, roll)
            elif  2.2 <= diff < 2.6:
                reward -= 0.5
                self.stable_counter = 0
            else:
                reward -= 4
                self.stable_counter = 0

            # For every foot that's not touching the ground, we take points

            """        
                for v in foot_contacts:
                if v == 0: reward -= 0.5
                elif v == 1: reward += 1
                else: print("NON-READABLE FOOT SENSOR VALUE! ", v)
            """
           
                
            """
             Following the mantis tutorial, after reading the .wbt
              file values for the hinge position:
               - if the "elbow" hinges were to be perfectly bent/balanced, 
               its angle would be [ ~ -2.4121293759260714 rad -> ~ -138.2 deg ]
            
             Therefore, the lower this negative number is, the tighter the 
              robot closes its arm.
            
             In order to, again, respect a threshold as it was done in
              the height check.

             However, the correct sensor placement is being blocked by internal 
             Webots processes. This makes it impossible to measure.
            """

            """
            # Acceptable arm position (hinge safety and correct execution of task)
            aC, aF, aT = 0.25, 0.20, 0.05  # perfect value amplitudes
            dC, dF, dT = 0.60, 0.80, -2.40  # offsets (centers)
            minC, maxC = dC - aC, dC + aC
            minF, maxF = dF - aF, dF + aF
            minT, maxT = dT - aT, dT + aT

            # Shoulders are not as important as elbow bend, so their reward is smaller
            for i in range(18):
                if i < 6:
                    if minC <= joint_sensors[i] <= maxC: 
                        # print("Considers perfect interval for base!")
                        reward += 0.2
                    else: reward -= 1

                elif i < 12:
                    if minF <= joint_sensors[i] <= maxF: 
                        reward += 0.2
                    else: reward -= 1

                elif i < 18:
                    if minT <= joint_sensors[i] <= maxT:
                        print("Considers perfect interval for elbow!")
                        reward += 0.2
                    else: reward -= 1
            """

            if self.task == 'walk':
                # Walking forward is -x (negative) and not moving on y
                # so if I take points for negative x plus y, I give points
                # when it walks forward in a straight line
                self.cur_dist = robot_pose[0]
                if self.checkTilt(pitch, roll) >= 1: reward -= 2*(robot_pose[0] + robot_pose[0])
                else: reward -= 1


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
        #print("Action sent by PPO: ", action)
        self.conn.sendall(json.dumps(action.tolist()).encode('utf-8'))
        #self.conn.sendall(json.dumps((np.zeros(30)).tolist()).encode('utf-8'))

        # Pulls readings after actions
        data = self.conn.recv(4096)
        obs = json.loads(data.decode('utf-8'))

        # Transforms into numpy array for efficiency of reward calculations
        observation = np.array(
            obs['joint_sensors'] + obs['imu'] + obs['foot_contacts'] + obs['com'],
            dtype=np.float32
        )


        #print("Joint sensors:", len(obs['joint_sensors']))
        """
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
        initial_obs = np.zeros(OBS_SPACE_SIZE, dtype=np.float32)
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