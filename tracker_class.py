# imports
import cv2
import moviepy.editor as mved
import pygame
# from moviepy.editor import *
import moviepy.video.fx.crop as moviefxcrop
import argparse
import numpy as np
from scipy import signal

#shape = []
#x_array = []
#y_array = []


class tracker:
    def __init__(self):
        # we need the [x,y] coordinates at time t
        self.x = 0
        self.y = 0

    def advance_by_frame(self, curr_frame, nxt_frame, corr_len=3):
        # definitions
        corr = 0
        search_radios = 1
        # we take the (corr_len X corr_len) matrix from curr_frame
        base = curr_frame[self.y - ((corr_len - 1) / 2):self.y + ((corr_len - 1) / 2),
               self.x - ((corr_len - 1) / 2):self.x + ((corr_len - 1) / 2)]
        base = base[:, :, 1]
        # we take a 3X3 area around the base coordinate pad it and check correlation.
        # if the max correlation in the area isn't good enough we increase the search radius until 10 pixels.
        for search_radios in range(1, 10):
            roi_mat = nxt_frame[self.y - ((corr_len - search_radios) / 2):self.y + ((corr_len - search_radios) / 2),
                      self.x - ((corr_len - search_radios) / 2):self.x + ((corr_len - search_radios) / 2)]
            roi_mat = np.asarray(roi_mat, dtype=np.int64)
            roi_mat = roi_mat[:, :, 1]
            pad_mat = np.pad(roi_mat, [(2, 2), (2, 2)], mode='constant', constant_values=0)
            corr = signal.correlate2d(base, pad_mat)
            if np.amax(corr) >= 0.8:
                break
        coordinates = np.where(corr == np.amax(corr))
        self.x = self.x + coordinates[0][0] - search_radios
        self.y = self.y + coordinates[1][0] - search_radios
        return


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


def Cut_frame_by_trackers(tracker_ls, frame):
    # This function get a list of 4 trackers and a frame and return the mask with 0 at any pixel
    # which isn't in the area defined by those pixels and 1 otherwise.

    Get_min_mask_array_shape(tracker_ls)
    t_ls = [tracker(), tracker(), tracker(), tracker()]
    #todo

    mask1 = Get_Mask_frame_above_line(tracker_ls[0], tracker_ls[1], frame.shape[0:2])
    mask2 = Get_Mask_frame_above_line(tracker_ls[0], tracker_ls[2], frame.shape[0:2])
    mask3 = Get_Mask_frame_above_line(tracker_ls[1], tracker_ls[3], frame.shape[0:2])
    mask4 = Get_Mask_frame_above_line(tracker_ls[2], tracker_ls[3], frame.shape[0:2])

    # final_mask = mask1 & mask2 & !mask3 & !mask4
    final_mask = np.logical_and(np.logical_and(mask1, mask3), np.logical_not(np.logical_or(mask2, mask4)))

    return final_mask


def Get_Mask_frame_above_line(point1, point2, shape):
    # This function receives 2 points and a shape of an array and returns an array in the shape specified with
    # 1 at any point above the line created by the two points and 0 otherwise.

    # Setup_help_arrays(given_shape)

    # computing cross product

    cross_p = np.zeros(shape)
    v1x = point2.x - point1.x
    v1y = point2.y - point1.y

    for i in range(shape[0]):
        for j in range(shape[1]):
            cross_p[i, j] = v1x * (point2.y - j) - v1y * (point2.x - i)

    mask = cross_p >= 0
    return mask


def Get_min_mask_array_shape(tracker_ls):
    min_x=min(tracker_ls[0].x,tracker_ls[1].x)
    max_x=max(tracker_ls[2].x,tracker_ls[3].x)
    min_y = min(tracker_ls[1].y, tracker_ls[3].y)
    max_y = max(tracker_ls[0].y, tracker_ls[2].y)
    return [min_x, max_x, min_y, max_y]