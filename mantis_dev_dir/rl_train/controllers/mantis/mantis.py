import socket
import math
import json
import random
import numpy as np
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

# ----------------------- Socket Setup ----------------------- #

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

# --------------------------------------------------------- #


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

    # Ending in C: controls leg (shoulder1?) forward-backward movements
    # Ending in F: controls body hinge (shoulder2?) up-down
    # Ending in T: controls arm hinge (elbow?) up-down

    # Anterior -> "face" of the robot
    # Posterior -> back of the robot

    MOTOR_EXPLANATION = [
        "RPC", "RPF", "RPT",  # Right Posterior Controls
        "RMC", "RMF", "RMT",  # Right Middle Controls
        "RAC", "RAF", "RAT",  # Right Anterior Controls
        "LPC", "LPF", "LPT",  # Left Posterior Controls
        "LMC", "LMF", "LMT",  # Left Middle Controls
        "LAC", "LAF", "LAT"   # Left Anterior Controls
    ]
    
    # For limit settings, we need to swap "lines" with "cols"
    MOTOR_NAMES = [
        "RPC", "RMC", "RAC", "LPC", "LMC", "LAC", # Base (shoulder1, front-backward) motors
        "RPF", "RMF", "RAF", "LPF", "LMF", "LAF", # Base (shoulder2, up_down) motors
        "RPT", "RMT", "RAT", "LPT", "LMT", "LAT"  # Hinge (elbow, up-down) motors
    ]

    motors, joint_sensors = [], []
    for name in MOTOR_NAMES:
        m = robot.getDevice(name)

        # Only elbows have sensors (manually added)
        if "T" in name:
            ps = robot.getDevice("ps_" + name)
            if ps:
                ps.enable(timestep)
                joint_sensors.append(ps)
        motors.append(m)

    
    # Integrity Limits for each motor (in radians)

    aC,aF,aT = 0.25, 0.20,  0.05           # perfect value amplitudes (too good)

    # custom values (simple integrity guide)
    # only keeping aC the same to avoid leg crossing
    aF *= 15
    aT *= 30


    dC,dF,dT = 0.60, 0.80, -2.40           # offsets (centers)

    minC, maxC = dC - aC, dC + aC
    minF, maxF = dF - aF, dF + aF
    minT, maxT = dT - aT, dT + aT


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


    # --------------- Robot and Elbows Frames --------------- #

    elbow_hinges_frames = []

    # Get robot node and translation field
    robot_translation_field = robot_node.getField("translation")

    # Get a specific hinge/joint node
    elbow_hinges = ["RPT", "RMT", "RAT", "LPT", "LMT", "LAT" ]
    for h in elbow_hinges:
        elbow_joint_node = robot.getFromDef(h+"_HINGE_JOINT")
        # print(h, ":", elbow_joint_node)
        if elbow_joint_node == None: continue
        elbow_translation_field = elbow_joint_node.getField("translation")
        elbow_hinges_frames.append(elbow_translation_field)

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
            # Normalizing motor input values (for safety and stability)
            #min_pos, max_pos = motors[i].getMinPosition(), motors[i].getMaxPosition()
            #pos = 0.5 * (action[i] + 1) * (max_pos - min_pos) + min_pos
            pos=action[i]

            # Normalizing to integrity limits based on "body part"

            # First 6 are C motors: shoulder forward-backward
            if i<6:
                pos = max(minC, pos)
                pos = min(maxC, pos)

            # The following 6 are F motors: shoulder up-down
            elif i<12:
                pos = max(minF, pos)
                pos = min(maxF, pos)

            # Final are T motors: elbow up-down
            elif i<18:
                pos = max(minT, pos)
                pos = min(maxT, pos)


            motors[i].setPosition(pos)
            #motors[i].setPosition(math.inf)
            #motors[i].setVelocity(pos)


                                            # ---------- Sensor Readings ---------- #

        # IMU
        roll, pitch, yaw = imu.getRollPitchYaw()
        # ax, ay, az = acc.getValues() if acc else (0.0, 0.0, 0.0)
        #acc_norm = math.sqrt(ax * ax + ay * ay + az * az)
        imu_values = [roll, pitch, yaw]

        # Read joint sensor angle values

        joint_values = []
        #print("---------------------------------------")
        for sensor in joint_sensors:
            #print("Positional sensor value: ", math.degrees(sensor.getValue()))
            if sensor: joint_values.append(sensor.getValue())
            else: joint_values.append(None)
        #print("---------------------------------------")


        # Difference between joint and robot heights
        joint_robot_hdiff = []
        robot_position = robot_translation_field.getSFVec3f()
        robot_height = robot_position[2]

        """
        for h in elbow_hinges_frames:
            print(h)
            hinge_position = h.getSFVec3f()
            hinge_height = hinge_position[2]
            hinge_robot_diff = hinge_height - robot_height
            joint_robot_hdiff.append(hinge_robot_diff)
        """
        

        # Read foot contact sensor values
        foot_values = [ts.getValue() for ts in feet]

        # Get center of mass approximation (using the robot's translation field)
        com = robot_node.getCenterOfMass()

        # Reads point cloud values
        point_cloud = lidar.getPointCloud()

        # We only want to see "forward": lidar points to the floor
        lidar_values = [p.x for p in point_cloud]


        # Collection of all relevant sensor/supervisor values
        observation = {
            # joint angles
            #"joint_robot_hdiff": joint_robot_hdiff,
            "joint_sensors" : joint_values,

            # roll, pitch and yaw
            "imu": imu_values, # 3 values

            # foot contact sensor values
            "foot_contacts": foot_values, # 6 values

            # center of mass (x,y,z)
            "com": com, # 3 values

            # robot distance to the ground
            "lidar": lidar_values # 3 values

        }

        """
        observation = {
            "joint_sensors": [random.uniform(-1.0, 1.0) for _ in range(6)],
            "imu": [random.uniform(-0.5, 0.5), random.uniform(-1, 1)],
            "foot_contacts": [random.randint(0, 1) for _ in range(6)],
            "com": [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
        }"""
        sock.sendall(json.dumps(observation).encode('utf-8'))


if __name__ == "__main__":
    main()
