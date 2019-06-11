#imports
import cv2
import moviepy.editor as mved
import pygame
#from moviepy.editor import *
import moviepy.video.fx.crop as moviefxcrop
import argparse


class tracker:
    def __init__(self):
        # we need the [x,y] coordinates at time t
        self.x=0
        self.y=0
    def advance_by_frame(self,curr_frame,nxt_frame,corr_len=3):
        # this code will not work if you have a coordinate too close to the edge


def create_4_trackers(x_left,x_right,y_top,y_bottom):
    # This function simply creates
    tracker_ls= [tracker(), tracker(), tracker(), tracker()]

    tracker_ls[0].x = x_left
    tracker_ls[1].x = x_left
    tracker_ls[2].x = x_right
    tracker_ls[3].x = x_right

    tracker_ls[0].y = y_top
    tracker_ls[1].y = y_bottom
    tracker_ls[2].y = y_top
    tracker_ls[3].y = y_bottom

    return tracker_ls