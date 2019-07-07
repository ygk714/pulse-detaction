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
    first_p0 = []
    curr_frame = []
    curr_gray = []
    diff_x = 0
    diff_y = 0
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
                p0 = get_points_to_track(curr_gray, face_roi)
                first_p0 = p0
            else:
                # Track
                p1, st, err = cv2.calcOpticalFlowPyrLK(prv_gray, curr_gray, p0, None, **lk_params)

                # Find the difference in the roi
                diff_x, diff_y = get_movement_from_trackers(first_p0, p1)
                # Get the rgb average in the forehead roi
                r_avg, g_avg, b_avg=get_forehead_rgb_vectors(curr_frame,forehead_roi,diff_x,diff_y)
                R_mat.append(r_avg)
                G_mat.append(g_avg)
                B_mat.append(b_avg)
                # Print frame + markers (debug)
                # if count % 10 == 1:
                #     print_frame_with_trackers(p0, p1, st, count, prv_frame, color, mask,diff_x,diff_y,forehead_roi)
                # advance the trackers
                p0 = (p1[st == 1]).reshape(-1, 1, 2)
            prv_frame = curr_frame
            prv_gray = curr_gray
            count += 1
            print (count)
        else:
            break

    print "hello"


def get_points_to_track(first_frame_gray, face_roi):
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    roi_gray = first_frame_gray[face_roi[2]:face_roi[3], face_roi[0]:face_roi[1]]
    # roi = first_frame[150:870, 1170:1670]

    p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)
    # normlize
    for i in range(len(p0)):
        p0[i][0][0] += face_roi[0]
        p0[i][0][1] += face_roi[2]
    return p0


def print_frame_with_trackers(p0, p1, st, counter, prv_frame, color, mask,diffx,diffy,forehead_roi):
    for i, (new, old) in enumerate(zip(p1[st == 1], p0[st == 1])):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        prv_frame = cv2.circle(prv_frame, (a, b), 5, color[i].tolist(), -1)
    prv_frame=cv2.rectangle(prv_frame,(forehead_roi[0]+diffx,forehead_roi[2]+diffy),(forehead_roi[1]+diffx,forehead_roi[3]+diffy),(0,255,0))
    cv2.imwrite("markers%d.jpg" % counter, prv_frame)  # save frame as JPEG file


def select_roi(first_frame):
    # The values are set to 'IMG_8900' values in the future define them differently
    y_min = 150
    y_max = 870
    x_min = 1170
    x_max = 1670
    face_roi = [x_min, x_max, y_min, y_max]
    y_min = 155
    y_max = 220
    x_min = 1225
    x_max = 1540
    forehead_roi = [x_min, x_max, y_min, y_max]
    return face_roi, forehead_roi


def get_movement_from_trackers(p0, p1):
    diff_x = []
    diff_y = []
    for i in range(len(p1)):
        diff_x.append(p1[i][0][0] - p0[i][0][0])
        diff_y.append(p1[i][0][1] - p0[i][0][1])
    avg_diff_x = np.around(np.average(diff_x)).astype(np.int)
    avg_diff_y = np.around(np.average(diff_y)).astype(np.int)
    return avg_diff_x, avg_diff_y


def get_forehead_rgb_vectors(frame, roi_forehead, diff_x, diff_y):
    forehead = frame[roi_forehead[2] + diff_y:roi_forehead[3] + diff_y, roi_forehead[0] + diff_x:roi_forehead[1] + diff_x]
    r_avg = np.average(forehead[:, :, 2])
    g_avg = np.average(forehead[:, :, 1])
    b_avg = np.average(forehead[:, :, 0])
    return r_avg,g_avg,b_avg


def band_passing(hf, lf, vector):
    from scipy.fftpack import fft
    y = fft(vector)
    j = 0
    for i in range(len(y)):
        if hf<i or i<lf:
            y[i] = 0
        if i!=0:
            if y[i]>y[j]:
                j = i
    print(j)
    return y


if __name__ == "__main__":
    main()
