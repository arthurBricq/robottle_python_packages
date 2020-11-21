import numpy as np
import cv2

def get_map(map_as_array, n_w = 500, n_h = 500):
    """
    Returns the map as a 2D occupancy grid (numpy) from the long format of the map.
    """
    return np.array(map_as_array).reshape(n_w, n_h)


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
    rgb_img = cv2.cvtColor(binary*255, cv2.COLOR_GRAY2RGB)
    
    cntrs, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in cntrs if len(c) > N_points_min]
    contour = np.concatenate(contours).reshape(-1,2)
    rot_rect = cv2.minAreaRect(contour)
    
    if save_name: 
        # let's save the picture somewhere
        cv2.drawContours(rgb_img, contours, -1, (0,255,0), 2)
        box = np.int0(cv2.boxPoints(rot_rect)) # cv2.boxPoints(rect) for OpenCV 3.x
        cv2.drawContours(rgb_img,[box],0,(0,0,255),2)
    
    return rot_rect 
    

    
    
    

 # threshold for binary image 
 # median filter kernel size
 # number of points inside a contour to keep it 

    
# plotting happens here


    

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
