import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

from robottle_utils import map_utils

%matplotlib qt

# what goes on in this file ? 
# the goal of map analysis is to extract points on the SLAM map that corresponds to 
# real points in the map (the center of each zones)
# To do so, we must find the corners

def plot_image(img):
    plt.figure()
    plt.imshow(img, cmap = "binary")
    plt.show()

##################################
#%% find rocks or obstacles

robot_pos = [200, 200, 45]
zones = np.array(([ 63, 245], [298, 213], [45, 56], [280,  24]))

# compute the points delimiting the rocks zone
w = 0.25
p1 = w * zones[0] + (1 - w) * zones[2]
p2 = w * zones[1] + (1 - w) * zones[2]
p3 = w * zones[3] + (1 - w) * zones[2]

# get the position of the obstacle (b for bottle)
theta = robot_pos[2] / 57.3
dx_pixels = 10
b = [robot_pos[0] + dx_pixels * np.cos(theta), 
        robot_pos[1] + dx_pixels * np.sin(theta)] 

# compute distances 
v12 = p2 - p1
v1b = b - p1
d12_p = np.dot(v1b, v12) / np.linalg.norm(v12) # projection orthogonale
d1b = np.linalg.norm(v1b)
distance_to_rocks_1 = np.sqrt(d1b ** 2 - d12_p ** 2)

v32 = p2 - p3
v3b = b - p3
d32_p = np.dot(v3b, v32) / np.linalg.norm(v32) # projection orthogonale
d3b = np.linalg.norm(v3b)
distance_to_rocks_2 = np.sqrt(d3b ** 2 - d32_p ** 2)

# logic over distances
if min(distance_to_rocks_1, distance_to_rocks_2) < threshold_pixels:
    return True 
return False 

#%% let's make a little plot of this, for verification
plt.figure()
plt.plot(zones[:,0], zones[:,1], 'x')
plt.show()

################################
#%% Find neighboords of each point

zones = np.array(([ 63, 245], [398, 213], [45, 56], [380,  24]))

# for each point, find its neighors (can be done only one)
X = np.linalg.norm(zones.reshape(1,4,2) - zones.reshape(4,1,2), axis = 2)
Y = np.argsort(X, axis = -1)
neighbors = np.logical_or(Y == 1 , Y == 2)

# once we have this matrix definded, we compute an average between each point and it's neighbors

################################
#%% from zones, find target points

# the 4 zones 
zones = np.array(([ 63, 245], [398, 213], [45, 56], [380,  24]))

# for each zone, here is its 2 closest neighbours
neighbours = np.array([[False,  True,  True, False],
       [ True, False, False,  True],
       [ True, False, False,  True],
       [False,  True,  True, False]])

s = time.time()

# lambda to compute weighted average between 2 points
average_points = lambda p1, p2, w: w * p1 + (1-w) * p2

# this is what I want to vectorize
targets = []
target_weight = 0.8
for i, neighbour in enumerate(neighbours):
    zone = zones[i]
    points = zones[neighbour]
    p1 = average_points(zone, points[0], target_weight)
    p2 = average_points(zone, points[1], target_weight)
    target = average_points(p1, p2, 0.5)
    targets.append(target)
targets = np.array(targets)

e = time.time()
print(e-s)

# let's make a little plot of this, for verification
plt.figure()
plt.plot(zones[:,0], zones[:,1], 'x')
plt.plot(targets[:,0], targets[:,1], 'yx')
plt.show()


################################
#%% Find zones AFTER initial zones were found

# zones = (recycling, z2, z3, z4)

previous_zones = np.array(([ 63, 245], [398, 213], [45, 56], [380,  24]))
new_zones = np.array(([ 73, 255], [388, 223], [55, 46], [370,  34]))
np.random.shuffle(new_zones)

X = previous_zones.reshape(1, 4, 2) - new_zones.reshape(4, 1, 2)
idcs = (X * X).sum(axis=2).argmin(axis = 0)



###################################
#%% Make a nice plot with all the given information

occupancy = np.load("/home/arthur/dev/ros/data/maps/test11.npy")
# plot_image(occupancy)

robot_pos = np.array([150, 350, 1])

# obtain data from image processing
binary_grid = map_utils.filter_map(occupancy)
corners, area, contours  = map_utils.get_bounding_rect(binary_grid) 
zones = map_utils.get_initial_zones(corners, robot_pos[:2])

# make a nice plot with all this (code for the plotting function)
rgb_img = cv2.cvtColor(binary_grid*255, cv2.COLOR_GRAY2RGB)
# contours
cv2.drawContours(rgb_img, contours, -1, (0,255,0), 2)
# rectangle around them
cv2.drawContours(rgb_img,[corners],0,(0,0,255),2)
# position of the robot
cv2.circle(rgb_img, tuple(robot_pos[:2]), 5, (0,0,204), cv2.FILLED)
theta = robot_pos[2]
pt2 = robot_pos[:2] + 50 * np.array([np.cos(theta), np.sin(theta)])
cv2.arrowedLine(rgb_img, tuple(robot_pos[:2]),tuple(pt2.astype(int)), color = (0,0,204), thickness = 2)
# position of the 4 zones
colors = [(0, 128, 255), (0, 204, 0), (128, 128, 128), (153, 0, 0), ]
for i, z in enumerate(zones):
    cv2.circle(rgb_img, tuple(z), 15, colors[i], cv2.FILLED)


plt.imshow(rgb_img)
plt.show()



####################
#%% find zones at the begining from the robot

robot_position = np.array([250, 300])

# 1. find the 4 corners of the map (outer corners)
corners = map_utils.get_bounding_rect(occupancy) 

# 2. find closest and the further away from robot
# i stands for 'index'
distances = ((corners - robot_position) * (corners - robot_position)).sum(axis = 1)
i_r, i_p4 = distances.argmin(), distances.argmax()
r, p4  = corners[i_r], corners[i_p4]

# 3. use sign of the cross product to find which are zone 2 and zone 3
diag = p4 - r 
cross_products = np.cross(diag, corners - r)
i_p2, i_p3 = cross_products.argmin(), cross_products.argmax() 
p2, p3 = corners[i_p2], corners[i_p3]

(r, p2, p3, p4)

#######################
#%% threshold to get binary image

# according to 'cv2.findcontours' we want to have zero = 

occupancy = np.load("/home/arthur/dev/ros/data/maps/test11.npy")

#threshold = 90
#binary = np.uint8(occupancy > threshold)
#binary = cv2.medianBlur(binary, ksize=5)

binary_dilated, binary = map_utils.filter_map(occupancy)

# contour detection (WORKING but weak)

cntrs, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
rgb_img = cv2.cvtColor(binary_dilated*255, cv2.COLOR_GRAY2RGB)
contours = [c for c in cntrs if len(c) > 30]
cv2.drawContours(rgb_img, contours, -1, (0,255,0), 3)

# Find oriented rectangle (Working ! )

points = np.where(binary == 0)
X = np.array(points)
contour = np.concatenate(contours).reshape(-1,2)
rot_rect = cv2.minAreaRect(contour)
box = cv2.boxPoints(rot_rect) # cv2.boxPoints(rect) for OpenCV 3.x
box = np.int0(box)
cv2.drawContours(rgb_img,[box],0,(0,0,255),2)
plot_image(rgb_img)



#%% compute principal axis and plot them (not working)

points = np.where(binary == 0)
X = np.array(points)
origin = X.mean(axis = 1)
# fake points for verification
# X = np.random.multivariate_normal(origin, [[500, 0], [0, 3000]], 300).T
cov = np.cov(X)
vals, vecs = np.linalg.eig(cov)
print(vecs)

# and make a plot
scales = 4 * vals.max() / vals
plt.scatter(X[0, :], X[1, :])
plt.quiver(*origin, *vecs[0], scale=scales[0], color='r')
plt.quiver(*origin, *vecs[1], scale=scales[1], color = 'b')
plt.legend(["points","v1","v2"])
plt.xlim(0, 500)
plt.ylim(0, 500)


#%% py shi tomasi corner detection (not working)

img = occupancy
rgb_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(img,25,0.01,10)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(rgb_img,(x,y),3,255,-1)
plt.imshow(rgb_img),plt.show()

#%% morpological operations (not working)

n = 5
kernel = np.ones((n,n),np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
plot_image(closing)

#%% harrys filter for corner detection (not working)

dst = cv2.cornerHarris(np.float32(occupancy),2,3,0.04)
# result iabsdiffs dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img = np.zeros(shape=binary.shape)
img[dst>0.7*dst.max()]=255
plot_image(img)


#%% FAST corner detection (not working)

img = occupancy
fast = cv2.FastFeatureDetector_create(threshold = 5)
# find and draw the keypoints
kp = fast.detect(binary,None)
img2 = cv2.drawKeypoints(binary, kp, outImage = None, color=(255,0,0))

plot_image(img2)

