# Python code to open a video stream and then take pictures when pressing 'i'. 

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


def show_camera():
    """
    This function opens a video stream from CSI camera, 
    opens a new window to display the stream,
    takes picture when 'i' is pressed, 
    and quit when 'esc' is pressed. 
    """
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    i = 0 
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            cv2.imshow("CSI Camera", img)
            keyCode = cv2.waitKey(30) & 0xFF
            if keyCode == 105: # 'i'
                name = output + str(i) + ".jpg"
                print("Writing image at : ", name)
                cv2.imwrite(name, img)
                i += 1
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    show_camera()
