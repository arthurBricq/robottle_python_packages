import matplotlib.pyplot as plt
import numpy as np
import cv2

def get_map(map_as_array, n_w = 500, n_h = 500):
    """
    Returns the map as a 2D occupancy grid (numpy) from the long format of the map.
    """
    return np.array(map_as_array).reshape(n_w, n_h)

def plot_map(occupancy_grid):
    """Plot the map in the current matplotlib.pyplot.figure"""
    plt.imshow(occupancy_grid, cmap="binary")

def filter_map(occupancy_grid):
    """
    From the initial occupancy_grid (straight out of the SLAM) returns a new and cleaner version of it.
    """
    # TODO: mabye it is not necessary to do so, really depends on testing. 
    return 0

def make_nice_plot(binary_grid, save_name, robot_pos = None, contours = None, corners = [], zones = []):
    """Make a nice plot, depending on the given parameters

    Parameters
    binary_grid: binary map of the world
    save_name: where to save the img
    robot_pos: position of the robot (x,y,theta)
    contours: detected contours around obstacles
    corners: 4 corners of the bounding oriented rectangle
    """
    # create RGB image (we do want some color here !)
    rgb_img = cv2.cvtColor(binary_grid*255, cv2.COLOR_GRAY2RGB)
    if contours:
        cv2.drawContours(rgb_img, contours, -1, (0,255,0), 2)
    if len(corners):
        cv2.drawContours(rgb_img,[corners],0,(0,0,255),2)
    # save the image
    cv2.imwrite(save_name, rgb_img)

def get_bounding_rect(occupancy, threshold = 90, kernel_size = 5, N_points_min = 30, save_name = None):
    """Returns an oriented rectangle (x,y,w,h) which surrounds the map.

    Parameters
    occupancy ([[int]]): occupancy grid of the map
    threshold (int): binary threshold to discriminate obstacles from free space
    N_points_min (int): min number of points to keep a contour (will remove very small contours)
    save_name (string): name to save the figure at (if None provided, will not save)

    Returns
    rot_ret (x,y,w,h): the rectable that goes around the map
    """
    binary = np.uint8(occupancy > threshold)
    binary = cv2.medianBlur(binary, ksize = kernel_size)

    cntrs, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in cntrs if len(c) > N_points_min]
    if len(contours):
        contour = np.concatenate(contours).reshape(-1,2)
        rot_rect = cv2.minAreaRect(contour)
        box = np.int0(cv2.boxPoints(rot_rect)) # cv2.boxPoints(rect) for OpenCV 3.x
        if save_name:
            # let's save the picture somewhere
            make_nice_plot(binary, save_name, corners = box, contours = contours)
            # rgb_img = cv2.cvtColor(binary*255, cv2.COLOR_GRAY2RGB)
            # cv2.drawContours(rgb_img, contours, -1, (0,255,0), 2)
            # cv2.drawContours(rgb_img,[box],0,(0,0,255),2)
            # cv2.imwrite(save_name, rgb_img)
        return box

def get_zones(corners, robot_position):
    """Returns the outermost position of the zones (recyling, zone2, zone3, zone4)
    given the initial corners of the map and the initial robot position. 

    The function assumes that
    - the closest point of the robot is the recycling area
    - the further away is the zone 4 (the ramp) 
    - the zones 2 and 3 are at the right and at the left of the map
    """
    # 1. find points closest and the further away from robot
    # this gives recycling area and the ramp zone
    # i stands for 'index'
    distances = ((corners - robot_position) * (corners - robot_position)).sum(axis = 1)
    i_r, i_p4 = distances.argmin(), distances.argmax()
    r, p4  = corners[i_r], corners[i_p4]

    # 2. use sign of the cross product to find which are zone 2 and zone 3
    diag = p4 - r
    cross_products = np.cross(diag, corners - r)
    i_p2, i_p3 = cross_products.argmin(), cross_products.argmax()
    p2, p3 = corners[i_p2], corners[i_p3]

    return (r, p2, p3, p4)

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
