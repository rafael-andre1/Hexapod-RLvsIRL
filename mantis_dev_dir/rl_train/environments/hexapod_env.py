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

if is_port_in_use(5000):
    raise RuntimeError("Port 5000 is already in use. Please close existing process.")


class HexapodEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(18,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.get_initial_observation()),), dtype=np.float32)

        # Starts Webots with supervisor
        self.webots_process = subprocess.Popen([
            r"C:\Users\hasht\AppData\Local\Programs\Webots\msys64\mingw64\bin\webots.exe",
            "--stdout",
            "--no-rendering",
            "worlds/mantis.wbt"
        ])
        time.sleep(5)

        # Socket server (init)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(('127.0.0.1', 5000))
        self.sock.listen(1)
        self.conn, _ = self.sock.accept()

    def get_initial_observation(self):
        return np.zeros(30)  # Placeholder

    def step(self, action):
        print("Step received.")
        self.conn.sendall(json.dumps(action.tolist()).encode('utf-8'))
        data = self.conn.recv(4096)
        obs = json.loads(data.decode('utf-8'))

        observation = np.array(
            obs['joint_sensors'] + obs['imu'] + obs['foot_contacts'] + obs['com'],
            dtype=np.float32
        )

        # NEEDS REWARD DEFINITION
        reward = obs['com'][0]  # recompensa por mover-se no eixo x
        done = False  # ou define condições de término

        return observation, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.conn.close()
        self.sock.close()
        self.webots_process.terminate()
        time.sleep(2)
        return self.__init__().reset()

    def close(self):
        self.conn.close()
        self.sock.close()
        self.webots_process.terminate()
