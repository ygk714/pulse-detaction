import cv2
import moviepy.editor as mved
import pygame
import moviepy.video.fx.crop as moviefxcrop
import numpy as np
from matplotlib import pyplot as plt


def main():
    # Defining variables
    R_mat = []
    G_mat = []
    B_mat = []
    prv_frame = []
    prv_gray = []
    p0 = []
    curr_frame = []
    curr_gray = []
    mask = np.zeros_like(prv_frame)
    color = np.random.randint(0, 255, (100, 3))



    # Opening the input video
    vidcap = cv2.VideoCapture('IMG_8900.mp4')

    count = 0
    pos_frame = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    while True:
        success, curr_frame = vidcap.read()
        if success:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            if count == 0:  # Reading the first frame
                # Get points to track
                face_roi, forehead_roi = select_roi(curr_frame)
                p0 = get_points_to_track(curr_gray,face_roi)
            else:
                # Track
                p1, st, err = cv2.calcOpticalFlowPyrLK(prv_gray, curr_gray, p0, None, **lk_params)
                # Print frame + markers (debug)
                if count % 10 == 1:
                    print_frame_with_trackers(p0, p1, st, count, prv_frame, color, mask)
                # advance the points
                p0 = (p1[st == 1]).reshape(-1, 1, 2)
            prv_frame = curr_frame
            prv_gray = curr_gray
            count += 1
            print (count)
        else:
            break


def get_points_to_track(first_frame_gray,face_roi):
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    roi_gray = first_frame_gray[face_roi[2]:face_roi[3],face_roi[0]:face_roi[1]]
    # roi = first_frame[150:870, 1170:1670]

    p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)
    # normlize
    for i in range(len(p0)):
        p0[i][0][0] += face_roi[0]
        p0[i][0][1] += face_roi[2]
    return p0


def print_frame_with_trackers(p0, p1, st, counter, prv_frame, color, mask):
    for i, (new, old) in enumerate(zip(p1[st == 1], p0[st == 1])):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        prv_frame = cv2.circle(prv_frame, (a, b), 5, color[i].tolist(), -1)
        cv2.imwrite("markers%d.jpg" % counter, prv_frame)  # save frame as JPEG file

def select_roi(first_frame):
    # The values are set to 'IMG_8900' values in the future define them differently
    y_min = 150
    y_max = 870
    x_min = 1170
    x_max = 1670
    face_roi=[x_min,x_max,y_min,y_max]
    y_min = 155
    y_max = 220
    x_min = 1225
    x_max = 1540
    forehead_roi = [x_min, x_max, y_min, y_max]
    return face_roi,forehead_roi


# def get_movment_from_trackers(p0,p1):
#

if __name__ == "__main__":
    main()
