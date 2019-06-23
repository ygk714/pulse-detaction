# imports
import cv2
import moviepy.editor as mved
import pygame
# from moviepy.editor import *
import moviepy.video.fx.crop as moviefxcrop
import argparse
import numpy as np
from scipy import signal


class tracker:
    def __init__(self):
        # we need the [x,y] coordinates at time t
        self.x = 0
        self.y = 0

    def advance_by_frame(self, curr_frame, nxt_frame, radios=1):
        # definitions
        corr = 0
        search_radios = 1
        # we take the (corr_len X corr_len) matrix from curr_frame
        base = curr_frame[self.x - radios-1:self.x + radios, self.y - radios-1:self.y + radios, :]
        # base = base[:, :, 1]

        # we take a 3X3 area around the base coordinate pad it and check correlation.
        # if the max correlation in the area isn't good enough we increase the search radius until 10 pixels.
        for search_radios in range(1, 10):
            # roi_mat = nxt_frame[self.y - ((corr_len - search_radios) / 2):self.y + ((corr_len - search_radios) / 2),
            #           self.x - ((corr_len - search_radios) / 2):self.x + ((corr_len - search_radios) / 2)]
            s = search_radios + 1
            roi_mat = nxt_frame[self.x - s-1:self.x + s, self.y - s-1:self.y + s]
            roi_mat = np.asarray(roi_mat, dtype=np.int32)
            # RGB
            corr = [-1]
            for i in range(3):
                roi_mat_i = roi_mat[:, :, i]
                base_i = base[:, :, i]
                corr_i = signal.correlate2d(base_i, roi_mat_i,mode='valid')
                # can do better but for debugging
                if np.min(corr) == -1:
                    corr = np.zeros(corr_i.shape)
                corr += corr_i
            corr = np.divide(corr, (3 * 255 * 255))
            if np.amax(corr) >= 0.8:
                break
        if corr[search_radios-radios,search_radios-radios]==np.amax(corr):
            coordinates=[[search_radios-radios],[search_radios-radios]]
        else:
            coordinates = np.where(corr == np.amax(corr))
        # coordinates = np.where(corr == np.amax(corr))
        self.x = self.x + coordinates[0][0] - search_radios
        self.y = self.y + coordinates[1][0] - search_radios
        return

    def Cut_frame_by_tracker(self, frame, radius=9):
        # This function get a tracker and a frame and returns a (radius X radius) matrix around the frame
        offset = (radius - 1) / 2
        return frame[self.x - offset:self.x + offset, self.y - offset:self.y + offset]


def create_4_trackers(x_left, x_right, y_top, y_bottom):
    # This function simply creates
    tracker_ls = [tracker(), tracker(), tracker(), tracker()]

    tracker_ls[0].x = x_left
    tracker_ls[1].x = x_left
    tracker_ls[2].x = x_right
    tracker_ls[3].x = x_right

    tracker_ls[0].y = y_top
    tracker_ls[1].y = y_bottom
    tracker_ls[2].y = y_top
    tracker_ls[3].y = y_bottom

    return tracker_ls


# def Cut_frame_by_tracker(tracker, frame):
#     # This function get a tracker and a frame and returns a 9X9 matrix around the frame
#     return frame[tracker.x-3:tracker.x+3,tracker.y-3:tracker.y+3]

def Get_min_mask_array_shape(tracker_ls):
    min_x = min(tracker_ls[0].x, tracker_ls[1].x)
    max_x = max(tracker_ls[2].x, tracker_ls[3].x)
    min_y = min(tracker_ls[1].y, tracker_ls[3].y)
    max_y = max(tracker_ls[0].y, tracker_ls[2].y)
    return [min_x, max_x, min_y, max_y]
