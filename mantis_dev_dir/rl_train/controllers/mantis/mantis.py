import csv
import math
import os
import socket
import json
import random
import time

# Socket Initialization

"""
Este socket TCP (Transmission Control Protocol) permite troca de infomração bidirecional entre 2 ficheiros de python.
Para evitar correr a simulação inteira na UI do webots, utilizamos este socket para treinar apenas em python.
Essencialmente, o environment de gymnasium cria um servidor local, ao qual `controller.py` vai aceder.

# -------- Fluxo -------- #

A cada passo (step()):

 - Gym envia uma ação para o Webots pelo socket

 - Webots aplica valores nos actuators (motores)

 - Webots lê sensores e envia de volta a observação (estado atual)

 - Gym recebe valores, calcula reward
  
 - Com base na reward, gym decide o próximo passo
"""

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

HOST = '127.0.0.1'
PORT = 5000
MAX_RETRIES = 50
WAIT = 0.5  # seconds

connected = False
for attempt in range(MAX_RETRIES):
    try:
        sock.connect((HOST, PORT))
        connected = True
        print("[Controller] Connected to RL environment.")
        break
    except ConnectionRefusedError:
        print(f"[Controller] Waiting for RL environment... Attempt {attempt+1}")
        time.sleep(WAIT)

if not connected:
    print("[Controller] Could not connect to RL environment after several attempts. Exiting.")
    exit(1)


print("Entered RL Controller.")

#os.environ["WEBOTS_HOME"] = '/usr/local/webots'

from controller import Robot, Motor, InertialUnit, Supervisor, PositionSensor, TouchSensor

def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    is_supervisor = hasattr(robot, 'getSelf')

    # Define motor device names (following convention: side (R/L), position (A/M/P), joint (C/F/T))
    motor_names = [
        "RPC", "RPF", "RPT",
        "RMC", "RMF", "RMT",
        "RAC", "RAF", "RAT",
        "LPC", "LPF", "LPT",
        "LMC", "LMF", "LMT",
        "LAC", "LAF", "LAT"
    ]
    motors = [robot.getDevice(name) for name in motor_names]

    """

    # Retrieve joint position sensors (assumed names: "ps_<motor_name>")
    #joint_sensor_names = ["ps_" + name for name in motor_names]
    joint_sensor_names = ["RPC", "RAC", "RMC","LMC","LAC","LPC"]
    joint_sensors = [robot.getDevice(name) for name in joint_sensor_names]
    for sensor in joint_sensors:
        if sensor is not None:
            sensor.enable(timestep)

    

    # IMU device (ensure the correct name: update if necessary)
    imu = robot.getDevice("integral unit")
    if imu is not None:
        imu.enable(timestep)

    # Foot contact sensors (NOT YET IMPLEMENTED)
    foot_contact_names = ["foot_contact1", "foot_contact2", "foot_contact3",
                          "foot_contact4", "foot_contact5", "foot_contact6"]
    foot_contacts = [robot.getDevice(name) for name in foot_contact_names]
    for sensor in foot_contacts:
        if sensor is not None:
            sensor.enable(timestep)
            
    """

    # If using Supervisor mode for COM, get the COM via the "translation" field:
    if is_supervisor:
        robot_node = robot.getSelf()
        # Typically, the robot's position is stored in "translation"
        com_field = robot_node.getField("translation")

    """

    # Gait parameters
    f = 0.5  # frequency [Hz]

    # Amplitudes [rad]
    aC = 0.25  # base motors
    aF = 0.2   # shoulder motors
    aT = 0.05  # knee motors
    a = [aC, aF, -aT, -aC, -aF, aT, aC, aF, -aT, aC, -aF, aT, -aC, aF, -aT, aC, -aF, aT]

    # Phases [s]
    pC = 0.0
    pF = 2.0
    pT = 2.5
    p = [pC, pF, pT, pC, pF, pT, pC, pF, pT, pC, pF, pT, pC, pF, pT, pC, pF, pT]

    # Offsets [rad]
    dC = 0.6
    dF = 0.8
    dT = -2.4
    d = [-dC, dF, dT, 0.0, dF, dT, dC, dF, dT, dC, dF, dT, 0.0, dF, dT, -dC, dF, dT]
    
    """

    # Main simulation loop
    while robot.step(timestep) != -1:
        # Receive motor positions, command motors
        try:
            data = sock.recv(4096)
            if not data:
                print("[Controller] Socket fechado ou vazio.")
                break
            action = json.loads(data.decode('utf-8'))
        except json.JSONDecodeError:
            print("[Controller] JSON inválido, a tentar novamente...")
            continue

        for i in range(18):
            motors[i].setPosition(action[i])

        """
        # Read IMU values (roll, pitch, yaw)
        imu_values = [None, None, None]
        if imu is not None:
            imu_values = imu.getRollPitchYaw()

        # Read joint sensor values
        joint_values = []
        for sensor in joint_sensors:
            if sensor is not None:
                joint_values.append(sensor.getValue())
            else:
                joint_values.append(None)

        # Read foot contact sensor values
        foot_values = []
        for sensor in foot_contacts:
            if sensor is not None:
                foot_values.append(sensor.getValue())
            else:
                foot_values.append(None)

        # Get center of mass approximation (using the robot's translation field)
        com = [None, None, None]
        if is_supervisor:
            if com_field is not None:
                com = com_field.getSFVec3f()

        observation = {
            "joint_sensors": joint_values,
            "imu": imu_values,
            "foot_contacts": foot_values,
            "com": com
        }"""

        observation = {
            "joint_sensors": [random.uniform(-1.0, 1.0) for _ in range(18)],
            "imu": [random.uniform(-0.5, 0.5), random.uniform(-1, 1)],
            "foot_contacts": [random.randint(0, 1) for _ in range(6)],
            "com": [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
        }
        sock.sendall(json.dumps(observation).encode('utf-8'))


if __name__ == "__main__":
    main()
