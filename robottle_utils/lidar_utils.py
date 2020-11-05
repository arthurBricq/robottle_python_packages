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
        if distance > threshold:
            number_of_valid_points += 1 
        if number_of_valid_points >= n_points:
            break 
    i1 = i - number_of_valid_points + 2

    # find the last index 
    number_of_valid_points = 0
    for i, distance in enumerate(distances[-1::-1]):
        if distance > threshold:
            number_of_valid_points += 1 
        if number_of_valid_points >= n_points:
            break 
    i2 = len(distances) - (i - number_of_valid_points) - 1

    return (i1, i2)

