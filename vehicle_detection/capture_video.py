import os
import time
import numpy as np
import cv2


def open_cam_onboard(width, height):
    # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
    gst_str = ('nvcamerasrc ! '
               'video/x-raw(memory:NVMM), '
               'width=(int)1920, height=(int)1080, '
               'flip-method=0, '
               'format=(string)I420, framerate=(fraction)120/1 ! '
               'nvvidconv flip-method=2 ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


WIDTH = 1920
HEIGHT = 1080

# Define the video stream
cap = open_cam_onboard(WIDTH, HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/mnt/SD/output.avi', fourcc, 7.0, (WIDTH, HEIGHT))

# Number of iterations

while True:
    # Read frame from camera
    ret, image_np = cap.read()
    if ret is False:
        break
    # Display output
    # cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
    # cv2.imshow('object detection', image_np)
    out.write(image_np)

    # cv2.imshow('frame', image_np)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

cap.release()

out.release()

cv2.destroyAllWindows()

