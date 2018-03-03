#!/usr/bin/env python
from __future__ import print_function

# Enable CPU only
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# from imutils.video import FPS
import os
import cv2
import numpy as np
import time

from sign_naming import *
model_loc = get_network_model()
model_loc.load_weights('sign_localize/weights_best.h5')
model_det = pickle.load(open('sign_detect/model.pickle', 'rb')) 

detector = SignDetector((model_loc, model_det))


import argparse
parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('-p', '--pic', action='store_true', help='Process picture')
parser.add_argument('-c', '--cam', action='store_true', help='Camera source')
parser.add_argument('-f', '--filepath', action='store', help='Path to video file to process')
    
args = parser.parse_args()


def processPicrute(filepath):
    frame = cv2.imread(filepath)
    if frame is None:
        print('Failed to open file')
        return

    detector.process_naming(frame)

    cv2.imshow('frame',frame)
    cv2.waitKey(0)

def show_fps(fps):
    print('FPS: %g' % fps)

from utils.fps import *
fps = FPS(cb_time=1, cb_func=show_fps)

def processStream(source):

    cap = cv2.VideoCapture(source)
    if cap is None or not cap.isOpened():
        print('Failed to open file')
        exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,320);
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240);

    ret, frame = cap.read()
    if frame is None:
        print('Failed to read frame')
        exit(1)

    while(cap.isOpened()):        
        fps.start()

        ret, frame = cap.read()
        if frame is None:
            exit(1)
            
        # frame = cv2.resize(frame, (320, 240))
        detector.process_naming(frame)
        
        fps.stop()

        # cv2.imshow('frame',frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if args.pic and args.filepath:
        processPicrute(args.filepath)
        
    elif args.cam or args.filepath:
        if args.cam:
            source = '/dev/v4l/by-id/usb-046d_0825_CA00E440-video-index0'  # Logitech camera
        elif args.filepath:
            source = args.filepath

        processStream(source)
