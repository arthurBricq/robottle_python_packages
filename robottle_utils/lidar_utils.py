import numpy as np

def check_obstacle_ahead(distances, angles, save_index = None, length_to_check = 300, half_robot_width = 150):
    """Given LIDAR data, returns 'True' if there is an obstacle in front of the robot.

    The code looks for angles infront of the robot divided in 2 zones
    - the triangle in front of the robot, angles delimited by angle delta (zone 2)
    - the complementary part of this previous triangle from the rectangle in front of the robot (zone 1)

    For each of those angles, it computes the 'critical distance' at which there should not be an obtatscle.
    If there are, it counts the number of detected 'obstacles' (i.e. distances below critcal distance for this angle)
    and return a heuristic rule to distinguish standing bottles from real obstacles.
    """
    distances = np.array(distances)
    angles = np.array(angles)
    # compute angle to differentiate regions
    delta = np.arctan(half_robot_width / length_to_check) * 57.3

    # compute the indices for the 2 regions of interest
    idcs1 = np.logical_or(np.logical_and(angles <= 90, angles >= delta), np.logical_and(angles >= 270, angles <= (360 - delta)))
    idcs2 = np.logical_or(np.logical_and(angles >= 0, angles <= delta), np.logical_and(angles <= 360, angles >= (360 - delta)))

    # compute the critical ditance for those regions (set of angles)
    critical_distance_1 = half_robot_width / np.abs(np.sin(angles[idcs1] / 57.3))
    critical_distance_2 = length_to_check / np.abs(np.cos(angles[idcs2] / 57.3))

    # compare actual distances with critical distances
    obstacles1 = distances[idcs1] < critical_distance_1
    obstacles2 = distances[idcs2] < critical_distance_2
     
    # save for debug
    if save_index is not None:
        np.save("/home/arthur/dev/ros/data/lidar/angles_{}.npy".format(save_index), angles)
        np.save("/home/arthur/dev/ros/data/lidar/distances_{}.npy".format(save_index), distances)
    
    # return true depending on the obstacle detection
    print("lidar count: ", np.count_nonzero(obstacles1) + np.count_nonzero(obstacles2))
    to_return = (np.count_nonzero(obstacles1) + np.count_nonzero(obstacles2)) >= 8

    return to_return




def get_valid_lidar_range(distances, angles, threshold = 600, n_points = 6):
    """
    Given the 'distances' array of LIDAR, it returns the indexes 'i1' and 'i2' that must be taken into account.
    Indeed, the LIDAR also sees the Robot and we must remove those points from the analysis.

    It will not be called during final deployment, just is used for determining 'i1' and 'i2'.
    (Those values may change depending on the 3D layout)
    """

    # 1. First, check array sizes
    if len(distances) != len(angles):
        print("ERROR ARRAY SIZES: distances and angles must have the same size")
        return (0,len(distances)-1)

    # find the first index
    number_of_valid_points = 0
    for i, distance in enumerate(distances):
        if distance < threshold:
            number_of_valid_points += 1 
        if number_of_valid_points >= n_points:
            break 
    i1 = i - number_of_valid_points + 2

    # find the last index 
    number_of_valid_points = 0
    for i, distance in enumerate(distances[-1::-1]):
        if distance < threshold:
            number_of_valid_points += 1 
        if number_of_valid_points >= n_points:
            break 
    i2 = len(distances) - (i - number_of_valid_points) - 1

    return (i1, i2)

