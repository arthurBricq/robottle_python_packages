import numpy as np

def get_distance(p1, p2):
    """Return euclidean distance between 2 points"""
    return np.linalg.norm(p1-p2)

def get_path_orientation(path):
    """Returns the orientation of the beginging of path.
    Since we know that the path always starts at the robot, 
    we look at the 2 first points to determine the orientation."""
    direction = path[-2] - path[-1]
    return np.rad2deg(np.arctan2(direction[1], direction[0]))


