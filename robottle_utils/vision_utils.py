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

def get_angle_of_closest_bottle(detections, P = array([[-3.19228946e+01],
       [ 6.95338103e-02],
       [-2.71226544e-02],
       [-3.25254858e-05],
       [-4.34729830e-05],
       [ 7.66028795e-05],
       [ 1.47783810e-08],
       [ 5.86412498e-09],
       [ 1.05578507e-07],
       [-1.32650702e-07]])):
    """This function returns the angle to rotate in order to reach the next bottle
    towards the right of the robot.
    TODO It can also return None if no good candidates were found.

    PARAMETERS
    detections [(x,y,w,h)]
    """
    # find closest detection by checking center of y axis
    closest_detection = (0,0,0,0)
    for detection in detections:
        if detection[1] > closest_detection[1]:
            closest_detection = detection
    x,y,w,h = closest_detection
    y -= h/2

    # compute angle
    angle = P[6]*x**3 + P[7]*x**2*y + P[8]*x*y**2 + P[9]*y**3 + P[3]*x**2 + P[4]*x*y + P[5]*y**2 + P[1]*x + P[2]*y + P[0]

    # compute angle of detection: positive when bottle on right side
    return angle
