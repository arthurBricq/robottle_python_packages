import numpy as np

def get_map(map_as_array, n_w = 500, n_h = 500):
    """
    Returns the map as a 2D occupancy grid (numpy) from the long format of the map.
    """
    return np.array(map_as_array).reshape(n_w, n_h)

def inspect_line(occupancy_grid, robot_position, length, w=10, h=10):
    """
    Returns True if there is an obstacle in the line in front of the robot, of given length..

    Must be given
    - real dimensions of the map : w, h
    - dimensions of the map in pixels: n_x, n_y

    Restriction: it works only for a map with no distortion.

    TODO: length out of bounds
    """
    (x,y,theta) = robot_position
    theta = np.deg2rad(theta)
    n_w, n_h = occupancy_grid.shape

    robot_x = int(x / w * n_w)
    robot_y = int(y / h  * n_h)

    length_in_pixel = int(length / w * n_w)
    x_dir = robot_x
    y_dir = robot_y
    for _ in range(length_in_pixel):
        x_dir += np.cos(theta)
        y_dir += np.sin(theta)
        robot_dir_index = (int(round(y_dir)), int(round(x_dir)))
        if occupancy_grid[robot_dir_index] < 100:
            return True
    return False
