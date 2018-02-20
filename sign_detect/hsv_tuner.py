#!/usr/bin/env python
 # let know what language is used

import cv2
import numpy as np
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('filepath', action='store', help='Path to video file to process')

args = parser.parse_args()

window_name = 'traffic_light'
color_trackbar_Hmin = 'H__min'
color_trackbar_Hmax = 'H__max'
color_trackbar_Smin = 'S__min'
color_trackbar_Smax = 'S__max'
color_trackbar_Vmin = 'V_min'
color_trackbar_Vmax = 'V_max'

#red
# val_Hmin = 0
# val_Hmax = 21
# val_Smin = 155
# val_Smax = 255
# val_Vmin = 122
# val_Vmax = 190

# blue
val_Hmin = 91
val_Hmax = 127
val_Smin = 164 #75
val_Smax = 255
val_Vmin = 40
val_Vmax = 190

def main():
    cap = cv2.VideoCapture(args.filepath)
    if cap is None or not cap.isOpened():
        print('Failed to open file')
        exit(1)

    ret, frame = cap.read()
    if frame is None:
        print('Failed to read frame')
        exit(1)

    cv2.namedWindow(window_name)
    def nothing(x): pass
    cv2.createTrackbar(color_trackbar_Hmin, window_name, val_Hmin, 180, nothing)
    cv2.createTrackbar(color_trackbar_Hmax, window_name, val_Hmax, 180, nothing)
    cv2.createTrackbar(color_trackbar_Smin, window_name, val_Smin, 255, nothing)
    cv2.createTrackbar(color_trackbar_Smax, window_name, val_Smax, 255, nothing)
    cv2.createTrackbar(color_trackbar_Vmin, window_name, val_Vmin, 255, nothing)
    cv2.createTrackbar(color_trackbar_Vmax, window_name, val_Vmax, 255, nothing)

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            exit(1)

        work_frame = np.copy(frame)
        work_frame = cv2.resize(work_frame, (320, 240))
        work_frame_hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
        trackbar_Hmin_pos = cv2.getTrackbarPos(color_trackbar_Hmin, window_name)
        trackbar_Hmax_pos = cv2.getTrackbarPos(color_trackbar_Hmax, window_name)
        trackbar_Smin_pos = cv2.getTrackbarPos(color_trackbar_Smin, window_name)
        trackbar_Smax_pos = cv2.getTrackbarPos(color_trackbar_Smax, window_name)
        trackbar_Vmin_pos = cv2.getTrackbarPos(color_trackbar_Vmin, window_name)
        trackbar_Vmax_pos = cv2.getTrackbarPos(color_trackbar_Vmax, window_name)
        res_frame = cv2.inRange(work_frame_hsv, (trackbar_Hmin_pos, trackbar_Smin_pos, trackbar_Vmin_pos),\
                                                (trackbar_Hmax_pos, trackbar_Smax_pos, trackbar_Vmax_pos))
        frame_inRange = np.copy(res_frame)
        ksize = 3
        open_kernel = np.ones((ksize, ksize), np.uint8)
        close_kernel = np.ones((ksize, ksize), np.uint8)

        res_frame = cv2.morphologyEx(res_frame, cv2.MORPH_OPEN, open_kernel) # erosion followed by dilation

        # res_frame = cv2.morphologyEx(res_frame, cv2.MORPH_CLOSE, close_kernel) # dilation followed by erosion

        res_frame = np.hstack((work_frame, 
                               cv2.cvtColor(frame_inRange, cv2.COLOR_GRAY2BGR),
                               cv2.cvtColor(res_frame, cv2.COLOR_GRAY2BGR)))

        cv2.imshow(window_name, res_frame)

        key = cv2.waitKey(0)
        if key == ord('x'): break
        if key == ord(' '): continue

if __name__ == '__main__':
    main()