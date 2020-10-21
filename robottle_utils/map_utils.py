import numpy as np

def get_map(map_as_array, width, height):
    """
    Returns the map as a 2D occupancy grid (numpy) from the long format of the map.
    """
    return np.array(map_as_array).reshape(width, height)
