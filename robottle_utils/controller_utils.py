import numpy as np

ROTATION_SPEED = 19 # [deg / sec]

def get_distance(p1, p2):
    """Return euclidean distance between 2 points"""
    return np.linalg.norm(p1-p2)

def get_path_orientation(path):
    """Returns the orientation of the beginging of path.
    Since we know that the path always starts at the robot, 
    we look at the 2 first points to determine the orientation."""
    direction = path[-2] - path[-1]
    return np.rad2deg(np.arctan2(direction[1], direction[0]))

def get_rotation_time(angle_to_reach):
    """Given an angle to rotate, estimate the required rotation time in seconds
    to reach this angle.
    TODO: tune this function
    """
    return angle_to_reach / ROTATION_SPEED

def is_obstacle_a_rock(robot_pos, zones, threshold_pixels = 30):
    """Given the position of the robot and the zones, 
    it will do the following steps
    - find the points delimiting the rocks zone
    - compute the minimum distance between the estimated position of the bottle and the line of rocks

    TODO
    - tune dx_pixels and threshold_pixels
    """
    if zones is None or not len(zones):
        return False

    # compute the points delimiting the rocks zone
    w = 0.25
    zones = np.array(zones)
    print(zones)
    p1 = w * zones[0] + (1 - w) * zones[2]
    p2 = w * zones[1] + (1 - w) * zones[2]
    p3 = w * zones[3] + (1 - w) * zones[2]

    # get the position of the obstacle (b for bottle)
    theta = robot_pos[2] / 57.3
    # TODO
    dx_pixels = 10
    b = [robot_pos[0] + dx_pixels * np.cos(theta),
            robot_pos[1] + dx_pixels * np.sin(theta)]
    print(b, robot_pos)

    # compute distances
    v12 = p2 - p1
    v1b = b - p1
    d12_p = np.dot(v1b, v12) / np.linalg.norm(v12) # projection orthogonale
    d1b = np.linalg.norm(v1b)
    distance_to_rocks_1 = np.sqrt(d1b ** 2 - d12_p ** 2)

    v32 = p2 - p3
    v3b = b - p3
    d32_p = np.dot(v3b, v32) / np.linalg.norm(v32) # projection orthogonale
    d3b = np.linalg.norm(v3b)
    distance_to_rocks_2 = np.sqrt(d3b ** 2 - d32_p ** 2)

    # compute the angle
    # TODO
    angle = 30

    # logic over distances
    if min(distance_to_rocks_1, distance_to_rocks_2) < threshold_pixels:
        return True, angle
    return False, None



