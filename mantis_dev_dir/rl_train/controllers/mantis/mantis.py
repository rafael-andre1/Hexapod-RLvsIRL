import socket
import math
import json
import random
import time
from controller import Robot, Motor, InertialUnit, Supervisor, PositionSensor, TouchSensor

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

def main():
    robot = Supervisor()
    #robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    is_supervisor = hasattr(robot, 'getSelf')

    # --------------- Lidar Initialiaztion ---------------- #

    lidar = robot.getDevice("lidar")
    lidar.enable(timestep)
    lidar.enablePointCloud()


    # --------------- Motor and Joint Sensor Initialization --------------- #
    MOTOR_NAMES = [
        "RPC", "RPF", "RPT", "RMC", "RMF", "RMT", "RAC", "RAF", "RAT",
        "LPC", "LPF", "LPT", "LMC", "LMF", "LMT", "LAC", "LAF", "LAT"
    ]
    motors, joint_sensors = [], []
    for name in MOTOR_NAMES:
        m = robot.getDevice(name)
        ps = m.getPositionSensor()
        if ps:  ps.enable(timestep)
        motors.append(m)
        joint_sensors.append(ps)

    # --------------- IMU  --------------- #
    imu = robot.getDevice("inertial unit")
    #gyro = robot.getDevice("gyro")
    #acc = robot.getDevice("accelerometer")
    """
    for s in (imu, gyro, acc):
        if s:
            s.enable(timestep)"""
    imu.enable(timestep)

    # --------------- Foot Contact Sensors --------------- #
    FOOT_NAMES = ["LAS", "LMS", "LPS", "RAS", "RMS", "RPS"]
    feet = []
    for name in FOOT_NAMES:
        ts = robot.getDevice(name)
        if ts:
            ts.enable(timestep)
        else:
            print(f"[warn] foot sensor {name} not found")
        feet.append(ts)

    # --------------- Center Of Mass --------------- #
    if is_supervisor:
        robot_node = robot.getSelf()
    else: robot_node = None

                                # ---------------------------------------------------- #
                                # --------------- Main Simulation Loop --------------- #
                                # ---------------------------------------------------- #


    while robot.step(timestep) != -1:

        # Receive motor positions, apply actions
        data = sock.recv(4096)

        # ---------- Receive Action ---------- #
        try:
            if not data:
                print("[Controller] Socket closed or empty. Generating random action...")
                fake_obs = [random.uniform(-1.0, 1.0) for _ in range(29)]
                sock.sendall(json.dumps(fake_obs).encode('utf-8'))
                continue

            message = json.loads(data.decode('utf-8'))


            # ---------- Reset Pose ---------- #

            # TODO: Random resets
            if isinstance(message, dict) and message.get("command") == "reset":
                print("[Controller] Reset em curso...")
                if is_supervisor:
                    robot.simulationReset()
                    """
                    robot_node.getField("translation").setSFVec3f([0, 0, 0.02])
                    robot_node.getField("rotation").setSFRotation([1, 0, 0, 1.57])
                    for motor in motors: motor.setPosition(float('0'))
                    """

                sock.sendall((json.dumps({"status": "reset_complete"}) + "\n").encode("utf-8"))
                continue  

        # ---------- Apply Action ---------- #    

            action = message

        except json.JSONDecodeError as e:
            print("[Controller] JSON inválido, a tentar novamente...")
            print(e)
            print(data)
            continue

        for i in range(18):
            motors[i].setPosition(action[i])

        # ---------- Sensor Readings ---------- #

        # IMU
        roll, p, y = imu.getRollPitchYaw() if imu else (0.0, 0.0, 0.0)
        # ax, ay, az = acc.getValues() if acc else (0.0, 0.0, 0.0)
        #acc_norm = math.sqrt(ax * ax + ay * ay + az * az)
        #imu_values = [roll, acc_norm]

        # Read joint sensor position values
        joint_values = []
        for sensor in joint_sensors:
            if sensor is not None: joint_values.append(sensor.getValue())
            else: joint_values.append(None)

        # Read foot contact sensor values
        foot_values = [ts.getValue() for ts in feet]

        # Get center of mass approximation (using the robot's translation field)
        com = robot_node.getCenterOfMass()

        # TODO: Can't send via json!!!
        point_cloud = lidar.getPointCloud()

        # We only want to see "forward", lidar points to the floor
        lidar_values = [p.x for p in point_cloud]
        print(lidar_values)


        # TODO: Currently random for joint sensors!!!
        observation = {
            # joint positions
            #"joint_sensors": joint_values,
            "joint_sensors": [random.uniform(-1.0, 1.0) for _ in range(18)],

            # roll and acceleration
            # "imu": imu_values,
            "imu": [random.uniform(-0.5, 0.5), random.uniform(-1, 1)],

            # foot contact sensor values
            "foot_contacts": foot_values,

            # center of mass (x,y,z)
            "com": com,

            "lidar": lidar_values

        }

        """
        observation = {
            "joint_sensors": [random.uniform(-1.0, 1.0) for _ in range(18)],
            "imu": [random.uniform(-0.5, 0.5), random.uniform(-1, 1)],
            "foot_contacts": [random.randint(0, 1) for _ in range(6)],
            "com": [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
        }"""
        sock.sendall(json.dumps(observation).encode('utf-8'))


if __name__ == "__main__":
    main()
