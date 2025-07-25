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

# Integrity Limits Velocity Cap
def cappedVelocity(cur_pos, vel, max_pos, min_pos, timestep=1000):
    # Max and Min velocity values based on previously defined integrity limits
    v_max, v_min = ((max_pos - cur_pos) / timestep), ((min_pos - cur_pos) / timestep)

    # first, prevent from going over the maximum value
    v_capped = min(v_max, vel)

    # finally, prevent from going under the minimum value
    return max(v_min, v_capped)

# Velocity for Expert Mode (GAIL) (0.8 -> overall best results)
def getExpertVelocity(motor_name):
    # Average velocities computed over the time steps (from 0.02s to 0.05s)
    # of the first few actions of the expert

    # These are just for smoothness of movement, therefore they must be abs()
    avg_velocities = {
        "RPC": 0.7814,
        "RPF": -0.3139,
        "RPT": 0.1340,
        "RMC": -0.7814,
        "RMF": 0.3139,
        "RMT": -0.1340,
        "RAC": 0.7814,
        "RAF": -0.3139,
        "RAT": 0.1340,
        "LPC": 0.7814,
        "LPF": 0.3139,
        "LPT": -0.1340,
        "LMC": -0.7814,
        "LMF": -0.3139,
        "LMT": 0.1340,
        "LAC": 0.7814,
        "LAF": 0.3139,
        "LAT": -0.1340,
    }

    if motor_name not in avg_velocities:
        raise ValueError(f"Motor name '{motor_name}' not found.")

    return abs(avg_velocities[motor_name])

print("Entered RL Controller.")

#os.environ["WEBOTS_HOME"] = '/usr/local/webots'

def main():
    robot = Supervisor()
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

    MOTOR_NAMES = [
        "RPC", "RPF", "RPT",  # Right Posterior Controls
        "RMC", "RMF", "RMT",  # Right Middle Controls
        "RAC", "RAF", "RAT",  # Right Anterior Controls
        "LPC", "LPF", "LPT",  # Left Posterior Controls
        "LMC", "LMF", "LMT",  # Left Middle Controls
        "LAC", "LAF", "LAT"   # Left Anterior Controls
    ]
    
    # For limit settings, we need to swap "lines" with "cols"

    # Important to mention that angles are inverted in C hinges 
    # (one side moves forward when value is applied, other moves backwards)
    MOTOR_EXPLANATION = [
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

    # custom values (simple integrity guide, manually set)
    # only lowering aC to avoid leg crossing
    aC /= 3
    aF *= 5
    aT *= 30


    # dC,dF,dT = 0.60, 0.80, -2.40           # offsets (theoretically, centers, but not working)
    dC, dF, dT = 0, 0.8, -2.4

    MINC, MAXC = dC - aC, dC + aC
    MINF, MAXF = dF - aF, dF + aT
    MINT, MAXT = dT - aT, dT + aT

    # Normalizes Positions
    def normalizePos(motor_name, posit):
        if motor_name.endswith("C"):
            posit = max(MINC, posit)
            return min(MAXC, posit)
        elif motor_name.endswith("F"):
            posit = max(MINF, posit)
            return min(MAXF, posit)
        elif motor_name.endswith("T"):
            posit = max(MINT, posit)
            return min(MAXT, posit)
        print("WRONG NAMES!!!")

    # --------------- IMU  --------------- #
    imu = robot.getDevice("inertial unit")
    imu.enable(timestep)

    # --------------- Foot Contact Sensors --------------- #
    FOOT_NAMES = ["LAS", "LMS", "LPS", "RAS", "RMS", "RPS"]
    feet = []
    for name in FOOT_NAMES:
        ts = robot.getDevice(name)
        if ts: ts.enable(timestep)
        else: print(f"[warn] foot sensor {name} not found")
        feet.append(ts)

    # --------------- Useful for Translation --------------- #
    if is_supervisor: robot_node = robot.getSelf()

    # --------------- Robot and Elbows Frames --------------- #
    elbow_hinges_frames = []

    # Get a specific hinge/joint node
    elbow_hinges = ["RPT", "RMT", "RAT", "LPT", "LMT", "LAT" ]
    for h in elbow_hinges:
        elbow_joint_node = robot.getFromDef(h+"_HINGE_JOINT")
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
            if isinstance(message, dict) and message.get("command") == "reset":
                print("[Controller] Reset em curso...")
                if is_supervisor:
                    robot.simulationReset()
                sock.sendall((json.dumps({"status": "reset_complete"}) + "\n").encode("utf-8"))
                continue

        # ---------- Apply Action ---------- #

            action = message

        except json.JSONDecodeError as e:
            print("[Controller] JSON inválido, a tentar novamente...")
            print(e)
            print(data)
            continue

        # Fetch current position values
        joint_values = []
        #print("---------------------------------------")
        for motor in motors:
            joint_values.append(motor.getTargetPosition())

        """ IMPORTANT IF YOU OVERWRITE THE INITIAL POSITION!
        
        # Only needs to be done once if world is correctly saved
        # otherwise, DISABLE ACTIONS and run this once, then save
        
            for i in range(18):
            vel = action[i]
            cur_pos = joint_values[i]

            # Normalizing to integrity limits based on "body part"
            # ==
            # Blocking velocity from integrity violation levels

            # First 6 are C motors: shoulder forward-backward
            if i<6: cappedVelocity(cur_pos, vel, minC, maxC)

            # The following 6 are F motors: shoulder up-down
            elif i<12: cappedVelocity(cur_pos, vel, minF, maxF)

            # Final are T motors: elbow up-down
            elif i<18: cappedVelocity(cur_pos, vel, minT, maxT)

            motors[i].setPosition(math.inf)
            motors[i].setVelocity(vel)
        
        """

        # Actions
        for i in range(18):
            pos = action[i]

            # Normalizing to integrity limits based on "body part"
            pos = normalizePos(MOTOR_NAMES[i], pos)

            # Usually the most stable
            motors[i].setVelocity(0.75)
            motors[i].setPosition(pos)
            
                                            # ---------- Sensor Readings ---------- #

        # IMU
        roll, pitch, yaw = imu.getRollPitchYaw()
        imu_values = [roll, pitch, yaw]

        # Get the robot's position (also using the robot's translation field)
        robot_pose = list(robot_node.getField("translation").getSFVec3f())

        # Collection of all relevant sensor/supervisor values
        observation = {
            # roll, pitch and yaw
            "imu": imu_values, # 3 values

            # robot translation
            "robot_pose": robot_pose  # 3 values
        }
        sock.sendall(json.dumps(observation).encode('utf-8'))


if __name__ == "__main__":
    main()
