import numpy as np

ROTATION_SPEED = 18.3 # [deg / sec]

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

def lineseg_dists(p, a, b):
    """Cartesian distance from point to line segment

    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892

    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2)
        - b: np.array of shape (x, 2)
    """
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1])
                           .reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors  
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)

def is_obstacle_a_rock(robot_pos, zones, threshold_pixels = 12.5, dx_pixels = 10):
    """Given the position of the robot and the zones, 
    it will do the following steps
    - find the points delimiting the rocks zone
    - compute the minimum distance between the estimated position of the bottle and the line of rocks

    Parameters
    ----------
    robot pos (x_pix, y_pix, theta)
    zones 
    threshold_pixels (int): min distance between estimated bottle pos and the rocks lines. 
    dx_pixels (int): 10 is for 24 [cm] ahead of the robot, where bottle is. 

    """
    if zones is None or not len(zones):
        return False, None

    # compute the points delimiting the rocks zone
    w = 0.25
    zones = np.array(zones)
    p1 = w * zones[0] + (1 - w) * zones[2]
    p2 = w * zones[1] + (1 - w) * zones[2]
    p3 = w * zones[3] + (1 - w) * zones[2]

    # get the position of the obstacle (b for bottle)
    theta = robot_pos[2] / 57.3
    b = np.array([robot_pos[0] + dx_pixels * np.cos(theta),
            robot_pos[1] + dx_pixels * np.sin(theta)])

    # compute distances to rock lines
    points1 = np.array([p1, p2])
    points2 = np.array([p2, p3])
    distances = lineseg_dists(b, points1, points2)
    distance_to_rocks_1, distance_to_rocks_2 = distances[0], distances[1]
    print(distance_to_rocks_1, distance_to_rocks_2, min(distance_to_rocks_1, distance_to_rocks_2) < threshold_pixels)

    # logic over distances
    if min(distance_to_rocks_1, distance_to_rocks_2) < threshold_pixels:
        # compute the angle the robot needs to rotate
        if distance_to_rocks_1 < distance_to_rocks_2:
            # robot must go toward direction 90 degrees
            pass
        else: 
            # robot must go toward direction 
            pass

        angle = 30
        return True, angle
    return False, None


def angle_diff(theta1, theta2): 
    """
    Compute the angle difference for 2 angles [degrees]
    """
    diff = (theta1 - theta2 + 180) % 360 - 180
    return diff














