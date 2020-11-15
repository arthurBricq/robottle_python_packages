import os
import cv2

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
