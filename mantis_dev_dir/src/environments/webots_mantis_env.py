# src/environments/webots_mantis_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from controllers.mantis.mantis_controller import MantisController


class WebotsMantisEnv(gym.Env):
    """
    Ambiente Gymnasium para controlar o robô Mantis no Webots.
    """

    def __init__(self):
        super(WebotsMantisEnv, self).__init__()

        self.controller = MantisController(timestep=32)

        # Define o espaço de observação e ação
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self.current_step = 0
        self.max_steps = 1000

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Reset manual (reiniciar motores e sensores)
        obs = self.controller.get_observation()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        self.controller.apply_action(action)
        if not self.controller.step():
            return self._terminated_state()

        obs = self.controller.get_observation()
        reward = self._calculate_reward(obs, action)
        self.current_step += 1
        done = self.current_step >= self.max_steps

        return np.array(obs, dtype=np.float32), reward, done, False, {}

    def _calculate_reward(self, obs, action):
        """
        Define a função de recompensa personalizada.
        """
        # Exemplo: penaliza ação alta e recompensa posição próxima de zero
        reward = -np.sum(np.square(action))  # Penaliza movimentos grandes
        reward += -np.sum(np.abs(obs))       # Penaliza desvio dos sensores
        return reward

    def _terminated_state(self):
        return np.zeros(self.observation_space.shape, dtype=np.float32), -100.0, True, False, {}

    def render(self):
        pass

    def close(self):
        pass
