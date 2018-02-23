#!/usr/bin/env python2
from __future__ import print_function

# Enable CPU only
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# from imutils.video import FPS
import os
import cv2
import numpy as np
import time
from keras.models import Model, load_model, save_model

import nn.rects as R
from nn.net import *
import argparse

# from skvideo.io import VideoCapture

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('weights', action='store', help='Path to weights file')
parser.add_argument('-p', '--pic', action='store_true', help='Process picture')
parser.add_argument('-c', '--cam', action='store_true', help='Camera source')
parser.add_argument('-f', '--filepath', action='store', help='Path to video file to process')


args = parser.parse_args()

data_path = 'raw/'

try:
    xrange
except NameError:
    xrange = range

def process_naming(frame, model):
    input = preprocess_img(frame)
    mask = model.predict(np.array([input]))[0] * 255.

    # mask = masks[:,:,0]
    mask = cv2.resize(mask.astype('uint8'), (frame.shape[1], frame.shape[0]))
    im, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)

    cv2.drawContours(frame, contours, -1, (255, 255, 0), 4)

    # mask = masks[:,:,1]
    # mask = cv2.resize(mask.astype('uint8'), (frame.shape[1], frame.shape[0]))
    # im, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)

    # cv2.drawContours(frame, contours, -1, (0, 255, 255), 4)



def execute_model():
    start_time, end_time, full_time = 0, 0, 0
    frame_cntr   = 0
    measure_time = 1

    model = get_network_model()

    if args.pic:
        frame = cv2.imread(args.filepath)
        if frame is None:
            print('Failed to open file')
            exit(1)
        
        model.load_weights(args.weights)

        process_naming(frame, model)

        cv2.imshow('frame',frame)
        cv2.waitKey(0)
        exit(1)
    else:
        if args.cam:
            cam_src = '/dev/v4l/by-id/usb-046d_0825_CA00E440-video-index0'
        elif args.filapath:
            cam_src = args.filepath
        else:
            print('No source set')
            exit(1)

        cap = cv2.VideoCapture(cam_src)
        if cap is None or not cap.isOpened():
            print('Failed to open file')
            exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,320);
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240);

        ret, frame = cap.read()
        if frame is None:
            print('Failed to read frame')
            exit(1)

        model.load_weights(args.weights)

        while(cap.isOpened()):
            start_time = time.time()

            ret, frame = cap.read()
            if frame is None:
                exit(1)

            frame = cv2.resize(frame, (320, 240))
            process_naming(frame, model)
            
            end_time = time.time()

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            full_time   += end_time - start_time
            frame_cntr  += 1

            if full_time > measure_time:
                fps = float(frame_cntr) / full_time
                full_time  = 0
                frame_cntr = 0

                print(fps)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    execute_model()
