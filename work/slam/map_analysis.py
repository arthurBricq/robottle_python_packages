import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# what goes on in this file ? 
# the goal of map analysis is to extract points on the SLAM map that corresponds to 
# real points in the map (the center of each zones)
# To do so, we must find the corners

def plot_image(img):
    plt.imshow(img, cmap = "binary")
    plt.show()

#%% Read one of the map that was saved from ROS

occupancy = np.load("/home/arthur/dev/ros/data/maps/mercantour1240.npy")
plot_image(occupancy)




#%% Feature extration 1: all in one cell 


threshold = 90 # threshold for binary image 
kernel_size = 5 # median filter kernel size
N_points_min = 30 # number of points inside a contour to keep it 

start = time.time()


binary = np.uint8(occupancy > threshold)
binary = cv2.medianBlur(binary, ksize = kernel_size)
rgb_img = cv2.cvtColor(binary*255, cv2.COLOR_GRAY2RGB)

cntrs, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = [c for c in cntrs if len(c) > N_points_min]
contour = np.concatenate(contours).reshape(-1,2)
rot_rect = cv2.minAreaRect(contour)

end = time.time()
print("Elasped: ", end - start)


# plotting happens here

cv2.drawContours(rgb_img, contours, -1, (0,255,0), 2)
box = np.int0(cv2.boxPoints(rot_rect)) # cv2.boxPoints(rect) for OpenCV 3.x
cv2.drawContours(rgb_img,[box],0,(0,0,255),2)
plt.imshow(rgb_img)






#%% threshold to get binary image

# according to 'cv2.findcontours' we want to have zero = 

threshold = 90
binary = np.uint8(occupancy > threshold)
plot_image(binary)
binary = cv2.medianBlur(binary, ksize=5)
plot_image(binary)

#%% contour detection (WORKING but weak)

cntrs, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
rgb_img = cv2.cvtColor(binary*255, cv2.COLOR_GRAY2RGB)
contours = [c for c in cntrs if len(c) > 30]
cv2.drawContours(rgb_img, contours, -1, (0,255,0), 3)
plot_image(rgb_img)

#%% Find oriented rectangle (Working ! )

points = np.where(binary == 0)
X = np.array(points)
contour = np.concatenate(contours).reshape(-1,2)
rot_rect = cv2.minAreaRect(contour)
box = cv2.boxPoints(rot_rect) # cv2.boxPoints(rect) for OpenCV 3.x
box = np.int0(box)
cv2.drawContours(rgb_img,[box],0,(0,0,255),2)
plt.imshow(rgb_img)















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
rgb_img = cv2.cvtColor(binary, cv2.GRAY2RGB)
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

