from controller import Supervisor
import csv, math

# CORRECTLY READS AND WRITES RELEVANT DATA

FREQ_HZ  = 0.5
CSV_FILE = "expert_data.csv"

print("Mantis expert to .csv started.")

robot = Supervisor()
is_super = hasattr(robot, "getSelf")
robot_node = robot.getSelf() if is_super else None
timestep = int(robot.getBasicTimeStep())


# Important to mention that angles are inverted in C hinges
# (one side moves forward when value is applied, other moves backwards)
MOTOR_NAMES_SHOWCASE = [
    "RPC", "RMC", "RAC", "LPC", "LMC", "LAC", # Base (shoulder1, front-backward) motors
    "RPF", "RMF", "RAF", "LPF", "LMF", "LAF", # Base (shoulder2, up_down) motors
    "RPT", "RMT", "RAT", "LPT", "LMT", "LAT"  # Hinge (elbow, up-down) motors
]

MOTOR_NAMES = [
    "RPC","RPF","RPT",  "RMC","RMF","RMT",  "RAC","RAF","RAT",
    "LPC","LPF","LPT",  "LMC","LMF","LMT",  "LAC","LAF","LAT"
]

# Getting each motor's values
motors = []
[motors.append(robot.getDevice(name)) for name in MOTOR_NAMES]


# Getting IMU
imu  = robot.getDevice("inertial unit")
imu.enable(timestep)

# Foot contacts

FOOT_NAMES = ["LAS", "LMS", "LPS", "RAS", "RMS", "RPS"]
feet = []
for name in FOOT_NAMES:
    ts = robot.getDevice(name)
    if ts: ts.enable(timestep)
    else:print(f"[warn] foot sensor {name} not found")
    feet.append(ts)

foot_values = [ts.getValue() for ts in feet]

# Robot coordinates

robot_pose = list(robot_node.getField("translation").getSFVec3f())

# Perfect parameters
aC,aF,aT = 0.25, 0.20,  0.05           # amplitudes
dC,dF,dT = 0.60, 0.80, -2.40           # offsets
pC,pF,pT = 0.00, 2.00,  2.50           # phases
A = [ aC, aF,-aT,-aC,-aF, aT,  aC, aF,-aT,  aC,-aF, aT, -aC, aF,-aT,  aC,-aF, aT]
D = [-dC, dF, dT, 0.0, dF, dT,  dC, dF, dT,  dC, dF, dT, 0.0, dF, dT, -dC, dF, dT]
P = [ pC, pF, pT, pC, pF, pT,  pC, pF, pT,  pC, pF, pT, pC, pF, pT,  pC, pF, pT]

# CSV Header
# time | 18 commands | imu_roll | imu_pitch | imu_yaw | 6 feet | coordinates_x_y_z
header = (
    ["time"] +
    MOTOR_NAMES +
    ["imu_roll", "imu_pitch", "imu_yaw"] +
    FOOT_NAMES + ["mantis_x", "mantis_y", "mantis_z"]
)

with open(CSV_FILE, "w", newline="") as fp:
    writer = csv.writer(fp)
    writer.writerow(header)

    # ───────────────────── main loop ──────────────────────
    start_time = robot.getTime()
    while robot.step(timestep) != -1:
        if robot.getTime() - start_time >= 10.0: break
        t = robot.getTime()

        cmd = [
            A[i] * math.sin(2 * math.pi * FREQ_HZ * t + P[i]) + D[i]
            for i in range(18)
        ]
        for m, pos in zip(motors, cmd):
            m.setPosition(pos)

        # rotational motor values
        joint_values = []
        # print("---------------------------------------")
        for motor in motors: joint_values.append(motor.getTargetPosition())


        # IMU: roll + accel norm
        r, p, y = imu.getRollPitchYaw() if imu else (0.0, 0.0, 0.0)

        # feet
        foot_vals = [ts.getValue() if ts else None for ts in feet]

        # mantis coordinates
        robot_pose = list(robot_node.getField("translation").getSFVec3f())
        # ----- write one CSV row -----
        writer.writerow(
            [t]
            + joint_values
            + [r, p, y]
            + foot_vals
            + robot_pose
        )

print("Finished; data saved to", CSV_FILE)
