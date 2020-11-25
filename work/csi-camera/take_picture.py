# When called, it will take a picture
# and save it to the output directory provideWhen called, it will take a picture
# and save it to the output directory providedd

import cv2
import argparse

parser = argparse.ArgumentParser(description = "Take pictures when pressing 'i'")
parser.add_argument('--output', type=str, help="name of output folder")
args = parser.parse_args()
output = args.output

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

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
ret, frame = cap.read()
cv2.imwrite(output, frame)
cap.release()
