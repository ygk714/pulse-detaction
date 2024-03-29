# imports
import cv2
import moviepy.editor as mved
import pygame
# from moviepy.editor import *
import moviepy.video.fx.crop as moviefxcrop
import argparse
import tracker_class
import numpy as np

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

        # construct the argument parser and parse the arguments


vidcap = cv2.VideoCapture('IMG_8900.mp4')
vidclip = mved.VideoFileClip('IMG_8900.mp4')
success, image = vidcap.read()
count = 0
success = True
cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
success, image = vidcap.read()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Capture.JPG")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
image = cv2.imread("frame0.jpg")
clone = image.copy()
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
# cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(0) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:
    x_1 = min(refPt[0][0], refPt[1][0])
    x_2 = max(refPt[0][0], refPt[1][0])
    y_1 = min(refPt[0][1], refPt[1][1])
    y_2 = max(refPt[0][1], refPt[1][1])
    # now we have 4 points we want to track in the time domain
    tracker_ls = tracker_class.create_4_trackers(x_1, x_2, y_1, y_2)

    success, image1 = vidcap.read()
    success, image2 = vidcap.read()

    # now we save all the frames
    avg1_array = []
    avg2_array = []
    avg3_array = []

    curr_frame = []
    nxt_frame = []
    pos_frame = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
    while True:
        success, curr_frame = vidcap.read()
        if success:
            if (nxt_frame != []):
                for t in tracker_ls:
                    t.advance_by_frame(curr_frame, nxt_frame)
                    nxt_frame = curr_frame
                    # frames_array.append(nxt_frame)
                    # cv2.imshow("image",curr_frame)
                mask = tracker_class.Cut_frame_by_trackers(tracker_ls, curr_frame)
                avg1_array.append(np.average(curr_frame[:, :, 1], weights=mask))
                avg2_array.append(np.average(curr_frame[:, :, 2], weights=mask))
                avg3_array.append(np.average(curr_frame[:, :, 3], weights=mask))
            else:
                nxt_frame = curr_frame
            pos_frame = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
        else:
            break
    height, width, layers = curr_frame.shape
    size = (width, height)
    out = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 0.5, size)
    pygame.display.set_caption('forehead')

    cut = moviefxcrop.crop(vidclip, x1=x_1, x2=x_2, y1=y_1, y2=y_2)
    # cut.preview()
    # cv2.waitKey(0)


    pass
# close all open windows
cv2.destroyAllWindows()
