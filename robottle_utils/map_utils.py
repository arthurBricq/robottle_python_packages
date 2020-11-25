import matplotlib.pyplot as plt
import numpy as np
import cv2


def pos_to_gridpos(x, y, N_pixs = 500, L = 10):
    """Retuns the given position in meters to a grid position as a numpy array. 
    N_pixs is the number of pixels per side, and L is the actual lenght of 1 side.
    All values are returned as int (theta is in rad)
    The function assumes that the map is a square
    """
    return np.array([N_pixs * x / L, N_pixs * y / L]).astype(int)

def get_map(map_as_array, n_w = 500, n_h = 500):
    """
    Returns the map as a 2D occupancy grid (numpy) from the long format of the map.
    """
    return np.array(map_as_array).reshape(n_w, n_h)

def plot_map(occupancy_grid):
    """Plot the map in the current matplotlib.pyplot.figure"""
    plt.imshow(occupancy_grid, cmap="binary")


def make_nice_plot(binary_grid, save_name, robot_pos = [], theta = 0, contours = [], corners = [], zones = []):
    """Make a nice plot, depending on the given parameters
    and saves it at the desired destination

    Parameters
    binary_grid: binary map of the world
    save_name: where to save the img
    robot_pos: position of the robot (x,y)
    theta: orientation of the robot [deg]
    contours: detected contours around obstacles
    corners: 4 corners of the bounding oriented rectangle
    """
    # create RGB image (we do want some color here !)
    rgb_img = cv2.cvtColor(binary_grid*255, cv2.COLOR_GRAY2RGB)
    if len(robot_pos):
        print("Robot position : ", robot_pos)
        cv2.circle(rgb_img, tuple(robot_pos[:2]), 5, (0,0,204), cv2.FILLED)
        theta = np.deg2rad(theta)
        pt2 = robot_pos[:2] + 50 * np.array([np.cos(theta), np.sin(theta)])
        cv2.arrowedLine(rgb_img, tuple(robot_pos[:2]),tuple(pt2.astype(int)), color = (0,0,204), thickness = 2)
    if len(contours):
        cv2.drawContours(rgb_img, contours, -1, (0,255,0), 2)
    if len(corners):
        cv2.drawContours(rgb_img,[corners],0,(0,0,255),2)
    if len(zones):
        colors = [(0, 128, 255), (0, 204, 0), (128, 128, 128), (153, 0, 0), ]
        for i, z in enumerate(zones):
            cv2.circle(rgb_img, tuple(z), 15, colors[i], cv2.FILLED)

    # save the image
    cv2.imwrite(save_name, rgb_img)

def filter_map(occupancy, threshold = 90, kernel_size = 5):
    """From the initial occupancy_grid (straight out of the SLAM) returns a new and cleaner version of it.

    Parameters
    occupancy ([[int]]): occupancy grid given by SLAM as numpy array
    threshold (int): binary threshold to discriminate obstacles from free space
    kernel_size (int): kernel size to use for median filter.
    """
    binary = np.uint8(occupancy > threshold)
    binary = cv2.medianBlur(binary, ksize = kernel_size)
    return binary

def get_bounding_rect(binary_grid, N_points_min = 30, save_name = None):
    """Returns an oriented rectangle (x,y,w,h) which surrounds the map.

    Parameters
    binary ([[int]]): occupancy grid of the map as a binary map with only obstacles
    N_points_min (int): min number of points to keep a contour (will remove very small contours)
    save_name (string): name to save the figure at (if None provided, will not save)

    Returns
    rot_ret (x,y,w,h): the rectable that goes around the map
    """
    # find contours and only keep important ones
    cntrs, hierarchy = cv2.findContours(binary_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in cntrs if len(c) > N_points_min]
    if len(contours):
        # create an array of all the points in all contours
        contour = np.concatenate(contours).reshape(-1,2)
        # find a rotated rectangle around those points
        rot_rect = cv2.minAreaRect(contour)
        corners = np.int0(cv2.boxPoints(rot_rect)) # cv2.boxPoints(rect) for OpenCV 3.x
        if save_name:
            # let's save the picture somewhere
            make_nice_plot(binary_grid, save_name, corners = corners, contours = contours)
        return corners, contours

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
    i_p3, i_p2 = cross_products.argmin(), cross_products.argmax()
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
