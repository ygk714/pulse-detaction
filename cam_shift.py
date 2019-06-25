# imports
import cv2
import moviepy.editor as mved
import pygame
# from moviepy.editor import *
import moviepy.video.fx.crop as moviefxcrop
import argparse
import numpy as np
from scipy import signal



class camshift:
    def __init__(self, x1, x2, y1, y2, frame):
        hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0., 51., 89.)), np.array((17., 140., 255.)))
        self.roi = mask[min(x1, x2):max(x1, x2), min(y1, y2):max(y1, y2)]
        self.roi_hist = cv2.calcHist([self.roi], [0], None, [16], [0, 180])
        self.roi_hist = cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
        self.roi_box = (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))
        self.termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    def advance_by_frame(self, curr_frame, nxt_frame):
        nxt_frame_hsv = cv2.cvtColor(nxt_frame, cv2.COLOR_BGR2HSV)
        nxt_frame_hsv = cv2.inRange(nxt_frame_hsv, np.array((0., 51., 89.)), np.array((17., 140., 255.)))
        backProj = cv2.calcBackProject([nxt_frame_hsv], [0], self.roi_hist, [0, 180], 1)
        (r, self.roi_box) = cv2.CamShift(backProj, self.roi_box, self.termination)
