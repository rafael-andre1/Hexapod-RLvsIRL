# Observations Previously included:

# joint angles
# "joint_sensors" : joint_values, # 18 values

# foot contact sensor values
# "foot_contacts": foot_values, # 6 values

# center of mass (x,y,z)
# " com": com, # 3 values

# robot distance to the ground
# "lidar": lidar_values # 3 values (previous implementation)


"""
# Read joint sensor rad values (after applying velocity)
joint_values = []
#print("---------------------------------------")
for motor in motors:
    #print("Positional sensor value: ", math.degrees(sensor.getValue()))
    joint_values.append(motor.getTargetPosition())
#print("---------------------------------------")
"""

"""
for h in elbow_hinges_frames:
print(h)
hinge_position = h.getSFVec3f()
hinge_height = hinge_position[2]
hinge_robot_diff = hinge_height - robot_height
joint_robot_hdiff.append(hinge_robot_diff)
"""

"""
# Read foot contact sensor values
foot_values = [ts.getValue() for ts in feet]
"""

"""
# Get center of mass approximation (using the robot's translation field)
com = robot_node.getCenterOfMass()
"""

""" Unfortunately, sensors do not work.
# Reads point cloud values
point_cloud = lidar.getPointCloud()

# We only want to see "forward": lidar points to the floor
lidar_values = [p.x for p in point_cloud]
print(lidar_values)
"""

# For debugging

"""
observation = {
    "joint_sensors": [random.uniform(-1.0, 1.0) for _ in range(6)],
    "imu": [random.uniform(-0.5, 0.5), random.uniform(-1, 1)],
    "foot_contacts": [random.randint(0, 1) for _ in range(6)],
    "com": [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
}"""