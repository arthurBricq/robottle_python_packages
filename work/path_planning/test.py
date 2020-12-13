# this python code is how to interface with the 'path planning algorithm'
# INPUTS: 
# - map obtained from the SLAM and estimated pose of the robot
# - target position
# OUTPUT:
# - path to follow

import numpy as np
import matplotlib.pyplot as plt
from robottle_utils import map_utils
%matplotlib qt
%load_ext autoreload
%autoreload 2

from robottle_utils.rrt import RRT
from robottle_utils.rrt_star import RRTStar

import time 
import cv2

################
#%% Map Analysis : including map dilation


# 1. get a map
m = np.load("/home/arthur/dev/ros/data/maps/inhibit0.npy")

# 2. code that the controller does to find the binary map (with obstacles) 
binary = map_utils.filter_map(m)
N = 10
kernel = np.ones((N, N), np.uint8) 
binary_dilated = cv2.erode(binary, kernel)
# cv2.imshow("test", binary)
plt.imshow(binary)
plt.show()
plt.figure()
plt.imshow(binary_dilated)
plt.show()

robot_pos = np.array([100, 100])
theta = 0 
corners, area, contours = map_utils.get_bounding_rect(binary_dilated)
zones = map_utils.get_initial_zones(corners, robot_pos)
targets = map_utils.get_targets_from_zones(np.array(zones), target_weight = 0.7)

##############
#%% path planning
current_target = 2 
s = time.time()
rrt = RRTStar( 
        start = robot_pos,
        goal = targets[current_target],
        binary_obstacle = binary_dilated, 
        rand_area = [0, 500],
        expand_dis = 50,
        path_resolution = 1,
        goal_sample_rate = 5,
        max_iter = 500,
        )
path = rrt.planning(animation = False)
e = time.time()
print(e-s)

if True:
    rrt.draw_graph()
    plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
    plt.grid(True)
    plt.pause(0.01)  # Need for Mac
    plt.show()

###########
#%% plotting

img = map_utils.make_nice_plot(binary, "", robot_pos, theta, contours, corners, zones, np.array(path).astype(int), "hello there")
plt.imshow(img)
plt.show()



