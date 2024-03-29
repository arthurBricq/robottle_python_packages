import os
import cv2
import numpy as np

PIXELS_PER_DEGREE = 12.5    # pixels per degree on the x axis of an image
X_CENTER_PIXEL = 700        # pixel that corresponds to center of image in x direction

# private function to have access to the streaming
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=2,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def take_picture(save = False, name = ""):
    """Takes a picture with the CSI-camera and returns it"""
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    # cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_V4L)
    ret, frame = cap.read()
    cap.release()
    if save:
        # save the picture here
        folder = "/home/arthur/dev/ros/pictures/f1/"
        file_name = os.path.join(folder, name + '.jpg')
        print("Going to save at : ", file_name)
        print(cv2.imwrite(file_name ,frame))
    return frame



def save_picture(pixels, rows, cols, dim, name, folder):
    """Save the picture given using cv2, from the video input topic of ROS"""
    frame = np.reshape(pixels, (rows, cols, dim))
    # save the picture here
    file_name = os.path.join(folder, name + '.jpg')
    print("Going to save at : ", file_name)
    print(cv2.imwrite(file_name ,frame))
    return frame

def get_angle_of_detection(detection, P = np.array([[-3.41793628e+01],
       [ 7.18081916e-02],
       [-1.40474330e-02],
       [-4.51951707e-05],
       [-5.81118447e-06],
       [ 2.19569100e-05],
       [ 2.17476005e-08],
       [ 8.52839111e-09],
       [ 4.80946129e-08],
       [-5.69118159e-08]] )):
    """This function returns the angle to rotate in order to reach the next bottle
    towards the right of the robot.
    TODO It can also return None if no good candidates were found.

    PARAMETERS
    detections [(x,y,w,h)]
    """
    x,y,w,h = detection
    y += h/2

    # compute angle
    angle = P[6]*x**3 + P[7]*x**2*y + P[8]*x*y**2 + P[9]*y**3 + P[3]*x**2 + P[4]*x*y + P[5]*y**2 + P[1]*x + P[2]*y + P[0]

    # compute angle of detection: positive when bottle on right side
    return angle

def get_best_detections(detections, dim_h = 720, dim_w = 1280):
    """
    Given an array of detected bounding box, returns the one the robot
    must go toward.

    Parameters
    ----------
    detections ([(x,y,w,h,is_flipped)]), length must be greater than 1 !
    """
    # 1. find the best index
    shortest_distance = 10000
    for i in range(len(detections)):
        x,y,w,h,is_flipped = detections[i]
        distance = (dim_h - x - w/2)if is_flipped else (dim_h - y - h/2)
        if distance < shortest_distance:
            shortest_distance = distance
            best_index = i
    # 2. return the best detection
    x,y,w,h,is_flipped = detections[i]
    return (dim_w - y, x, h, w) if is_flipped else (x,y,w,h)







