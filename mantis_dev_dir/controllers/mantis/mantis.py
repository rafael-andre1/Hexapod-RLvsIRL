import csv
import math
import os

print("Entered Python controller!!!")

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

    # Retrieve joint position sensors (assumed names: "ps_<motor_name>")
    joint_sensor_names = ["ps_" + name for name in motor_names]
    joint_sensors = [robot.getDevice(name) for name in joint_sensor_names]
    for sensor in joint_sensors:
        if sensor is not None:
            sensor.enable(timestep)

    # IMU device (ensure the correct name: update if necessary)
    imu = robot.getDevice("integral unit")
    if imu is not None:
        imu.enable(timestep)

    # Foot contact sensors (assumed names)
    foot_contact_names = ["foot_contact1", "foot_contact2", "foot_contact3",
                          "foot_contact4", "foot_contact5", "foot_contact6"]
    foot_contacts = [robot.getDevice(name) for name in foot_contact_names]
    for sensor in foot_contacts:
        if sensor is not None:
            sensor.enable(timestep)

    # If using Supervisor mode for COM, get the COM via the "translation" field:
    if is_supervisor:
        robot_node = robot.getSelf()
        # Typically, the robot's position is stored in "translation"
        com_field = robot_node.getField("translation")

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

    # Open a CSV file to log all sensor data
    with open("expert_data.csv", mode='w', newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Build CSV header:
        header = ["time"]
        # Use the actual motor device names for clarity
        header += motor_names
        header += ["imu_roll", "imu_pitch", "imu_yaw"]
        header += ["joint_sensor{}".format(i+1) for i in range(len(joint_sensors))]
        header += foot_contact_names
        header += ["com_x", "com_y", "com_z"]
        csv_writer.writerow(header)

        # Main simulation loop
        while robot.step(timestep) != -1:
            current_time = robot.getTime()
            row = [current_time]

            # Compute motor positions, command motors, and log positions
            motor_positions = []
            for i in range(18):
                pos = a[i] * math.sin(2.0 * math.pi * f * current_time + p[i]) + d[i]
                motors[i].setPosition(pos)
                motor_positions.append(pos)
            row += motor_positions

            # Read IMU values (roll, pitch, yaw)
            imu_values = [None, None, None]
            if imu is not None:
                imu_values = imu.getRollPitchYaw()
            row += imu_values

            # Read joint sensor values
            joint_values = []
            for sensor in joint_sensors:
                if sensor is not None:
                    joint_values.append(sensor.getValue())
                else:
                    joint_values.append(None)
            row += joint_values

            # Read foot contact sensor values
            foot_values = []
            for sensor in foot_contacts:
                if sensor is not None:
                    foot_values.append(sensor.getValue())
                else:
                    foot_values.append(None)
            row += foot_values

            # Get center of mass approximation (using the robot's translation field)
            com = [None, None, None]
            if is_supervisor and com_field is not None:
                com = com_field.getSFVec3f()
            row += com

            csv_writer.writerow(row)

if __name__ == "__main__":
    main()
