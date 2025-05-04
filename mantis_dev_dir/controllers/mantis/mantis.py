# controllers/mantis/mantis_controller.py

from controller import Robot, Motor, InertialUnit, Supervisor, PositionSensor, TouchSensor

class MantisController:
    """
    Classe para encapsular o controlo do robô Mantis no Webots.
    Responsável por inicializar motores e sensores e aplicar ações.
    """

    def __init__(self, timestep=32):
        self.robot = Robot()
        self.timestep = timestep
        self.motors = []
        self.sensors = []
        self._initialize_devices()

    def _initialize_devices(self):
        """
        Inicializa os motores e sensores do robô.
        """
        motor_names = [f"motor{i}" for i in range(6)]  # Ajusta conforme teu robô
        for name in motor_names:
            motor = self.robot.getDevice(name)
            motor.setPosition(0.0)
            motor.setVelocity(0.0)
            self.motors.append(motor)

        sensor_names = [f"sensor{i}" for i in range(6)]  # Exemplo: sensores de posição
        for name in sensor_names:
            sensor = self.robot.getDevice(name)
            sensor.enable(self.timestep)
            self.sensors.append(sensor)

    def step(self):
        """
        Avança a simulação um passo. Deve ser chamado a cada iteração.
        Retorna False se a simulação for terminada.
        """
        return self.robot.step(self.timestep) != -1

    def get_observation(self):
        """
        Obtém observações do ambiente (e.g., leituras dos sensores).
        """
        return [sensor.getValue() for sensor in self.sensors]

    def apply_action(self, action):
        """
        Aplica ações aos motores do robô.
        """
        for i, motor in enumerate(self.motors):
            motor.setPosition(action[i])
