import matplotlib.pyplot as plt
import numpy as np
import cv2



### HELPER FUNCTIONS

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

### MAP FILTERING

def filter_map(occupancy, threshold = 90, median_kernel_size = 5, dilation_kernel_size = 10):
    """From the initial occupancy_grid (straight out of the SLAM) returns a new and cleaner version of it.
    It is possible to perform a binary dilation of the image, to inflate obstacles.

    Parameters
    occupancy ([[int]]): occupancy grid given by SLAM as numpy array
    threshold (int): binary threshold to discriminate obstacles from free space
    median_kernel_size (int): kernel size to use for median filter.
    dilation_kernel_size (int): kernel size to perform obstacles inflation
    """
    binary = np.uint8(occupancy > threshold)
    binary = cv2.medianBlur(binary, ksize = median_kernel_size)
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    binary_dilated = cv2.erode(binary, kernel)
    return binary_dilated, binary

def get_bounding_rect(binary_grid, N_points_min = 30, save_name = None):
    """Returns an oriented rectangle (x,y,w,h) which surrounds the map.

    Parameters
    binary ([[int]]): occupancy grid of the map as a binary map with only obstacles
    N_points_min (int): min number of points to keep a contour (will remove very small contours)
    save_name (string): name to save the figure at (if None provided, will not save)

    Returns
    corners (p1,p2,p3,p4): the 4 corners of the rectangles
    area (int): number of pixels inside the rectangle (used to select valid rectangle)
    contours ([[points]]): all the detected contours (used for plotting)
    """
    # find contours and only keep important ones
    cntrs, hierarchy = cv2.findContours(binary_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in cntrs if len(c) > N_points_min]
    if len(contours):
        # create an array of all the points in all contours
        contour = np.concatenate(contours).reshape(-1,2)
        # find a rotated rectangle around those points
        rot_rect = cv2.minAreaRect(contour)
        area = rot_rect[1][0] *  rot_rect[1][1] # rot_rect[1] = size = [w,h]
        corners = np.int0(cv2.boxPoints(rot_rect)) # cv2.boxPoints(rect) for OpenCV 3.x
        if save_name:
            # let's save the picture somewhere
            make_nice_plot(binary_grid, save_name, corners = corners, contours = contours)
        return corners, area, contours
    else:
        return None


### 'ZONES FUNCTION'

def get_initial_zones(corners, robot_position, closest_zone = 0):
    """Returns the outermost position of the zones (recyling, zone2, zone3, zone4)
    given the initial corners of the map and the initial robot position. 

    The function assumes that
    - the closest point of the robot is the recycling area
    - the further away is the zone 4 (the ramp) 
    - the zones 2 and 3 are at the right and at the left of the map

    DISCLAMER
    ---------
    parameter closest_zone must be either 0 or 2 (recycling or rocks)

    RETURNS
    --------
    zones (recycling, z2, z3, z4)
    """
    # 1. find points closest and the further away from robot
    # this gives recycling area and the ramp zone
    # i stands for 'index'
    distances = ((corners - robot_position) * (corners - robot_position)).sum(axis = 1)
    i_closesth, i_furthest = distances.argmin(), distances.argmax()
    closest, furthest = corners[i_closesth], corners[i_furthest]

    # 2. use sign of the cross product to find which are zone 2 and zone 3
    diag = furthest - closest
    cross_products = np.cross(diag, corners - closest)
    i_left, i_right = cross_products.argmin(), cross_products.argmax()
    right, left = corners[i_right], corners[i_left]

    if closest_zone == 0:
        return (closest, right, left, furthest)
    elif closest_zone == 2: 
        return (right, furthest, closest, left)

def get_zones_from_previous(corners, previous_zones):
    """Given the 4 corners found and the previous zones detected (assuming that those
    are the correct ones) it will order the new zones so that closest points are in the 
    same position.

    Returns
    zones (recycling, z2, z3, z4)
    """
    new_zones = np.array(corners)
    previous_zones = np.array(previous_zones)
    X = previous_zones.reshape(1, 4, 2) - new_zones.reshape(4, 1, 2)
    idcs = (X * X).sum(axis=2).argmin(axis = 0)
    return new_zones[idcs]

def are_new_zones_valid(new_zones, last_zones, threshold = 60):
    """Returns True if new zones are valid"""
    X = new_zones - last_zones
    dists = np.linalg.norm(X, axis = -1)
    if dists.max() > threshold: 
        return False
    else: 
        return True

# lambda to compute weighted average between 2 points
average_points = lambda p1, p2, w: w * p2 + (1-w) * p1

def get_targets_from_zones(zones): 
    """Given the computed zones, return target points where robot should go.
    Indeed, the 'zones' are the corner of the area and the robot can't go there.
    This function will select points that are within the real zone, and accessible for the robot. 

    Parameters
    zone (r, z2, z3, z4)
    target_weight (float in [0,1]): the smaller, the further away is the target from the corner.

    Requires
    -neighbours matrix must be defined
    -average_points lambda must be defined

    Returns 
    targets (r, z2, z3, z4): target points as defined above
    """
    targets = []
    # Targets = recycling, grass, in front of recycling, in front of ramp, rocks entry point, rocks exit point
    weigths1 = [0.20, 0.25, 0.30,  0.6, 0.6, 0.75]
    weigths2 = [0.20, 0.85, 0.30, 0.8, 0.2, 0.2]
    for w1, w2 in zip(weigths1, weigths2): 
        p1 = average_points(zones[0], zones[2], w1)
        p2 = average_points(zones[1], zones[3], w1)
        target = average_points(p1, p2, w2)
        targets.append(target)
    return np.array(targets).astype(int)


def get_random_area(zones):
    """Given the 4 zones, returns the random area [x_min, x_max, y_min, y_max] where to sample
    points for the path planner. 
    
    Parameters
    zones (numpy array of the 4 zones)
    """
    mins = zones.min(axis = 0)
    maxs = zones.max(axis = 0)
    return [mins[0], maxs[0], mins[1], maxs[1]]


### DEBUG FUNCTIONS

def make_nice_plot(binary_grid, save_name, robot_pos = [], theta = 0, contours = [], corners = [], zones = [], targets = [], path = [], text = ""):
    """Make a nice plot, depending on the given parameters
    and saves it at the desired destination
    Parameters
    binary_grid: binary map of the world
    save_name: where to save the img
    robot_pos: position of the robot (x,y)
    theta: orientation of the robot [deg]
    contours: detected contours around obstacles
    corners: 4 corners of the bounding oriented rectangle
    zones: 4 zones (order is the label)
    targerts: the 4 zones that were moved to be inside the arena
    path: list of points (best path so far)
    text: text to write on the image
    """
    # create RGB image (we do want some color here !)
    rgb_img = cv2.cvtColor(binary_grid*255, cv2.COLOR_GRAY2RGB)
    if len(contours):
        cv2.drawContours(rgb_img, contours, -1, (0,255,0), 2)
    if len(corners):
        cv2.drawContours(rgb_img,[corners],0,(0,0,255),2)
    if len(zones):
        colors = [(0, 128, 255), (0, 204, 0), (128, 128, 128), (153, 0, 0), ]
        for i, z in enumerate(zones):
            cv2.circle(rgb_img, tuple(z), 15, colors[i], cv2.FILLED)
    if len(targets):
        colors = [(0, 128, 255), (0, 204, 0), (153, 0, 0), (128, 128, 128), (200, 10, 100)]
        for i, z in enumerate(targets):
            cv2.circle(rgb_img, tuple(z), 5, colors[i], cv2.FILLED)
    if len(path):
        for i, _ in enumerate(path[:-1]):
            cv2.line(rgb_img, tuple(path[i]), tuple(path[i+1]), (0, 153, 51), 3) 
    if len(robot_pos):
        cv2.circle(rgb_img, tuple(robot_pos[:2]), 5, (0,0,204), cv2.FILLED)
        theta = np.deg2rad(theta)
        pt2 = robot_pos[:2] + 50 * np.array([np.cos(theta), np.sin(theta)])
        cv2.arrowedLine(rgb_img, tuple(robot_pos[:2]),tuple(pt2.astype(int)), color = (0,0,204), thickness = 2)
    if len(text): 
        y0, dy = 400, 40
        for i, line in enumerate(text.split('\n')):
            y = y0 + i*dy
            cv2.putText(rgb_img, line, (30, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 6)

    # save the image
    if save_name:
        cv2.imwrite(save_name, rgb_img)
    

    return rgb_img

def inspect_line(occupancy_grid, robot_position, length, w=10, h=10):
    """
    Returns True if there is an obstacle in the line in front of the robot, of
    given length.

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
