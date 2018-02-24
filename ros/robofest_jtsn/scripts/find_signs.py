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
from keras.models import Model, load_model, save_model

import sign_localize.rects as R
from sign_localize.net import *

import pickle
import sign_detect.ml_utils as mlu

import argparse
parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('-p', '--pic', action='store_true', help='Process picture')
parser.add_argument('-c', '--cam', action='store_true', help='Camera source')
parser.add_argument('-f', '--filepath', action='store', help='Path to video file to process')

args = parser.parse_args()

model_loc = get_network_model()
model_loc.load_weights('sign_localize/weights_best.h5')

model_det = pickle.load(open('sign_detect/model.pickle', 'rb'))

data_path = 'raw/'

try:
    xrange
except NameError:
    xrange = range

def process_naming(frame):
    input = preprocess_img(frame)
    mask = model_loc.predict(np.array([input]))[0] * 255.

    # mask = masks[:,:,0]
    mask = cv2.resize(mask.astype('uint8'), (frame.shape[1], frame.shape[0]))
    im, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    area    = 0
    pred    = None
    coords  = None

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)

        roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_RGB2GRAY)
        hog_features = mlu.get_hog_features(roi)
        y_pred = model_det.predict(hog_features.reshape(1, -1))[0]

        hull = cv2.convexHull(cnt)
        cv2.drawContours(frame, [hull], 0, (255, 255, 255), 4)

        if y_pred.title().lower() != 'negative':
            curr_area = cv2.contourArea(cnt)
            if curr_area > area:
                pred = y_pred.title().lower()
                area = curr_area
                coords = (x,y,w,h)
    
    ### Circle estimation

    # for con in contours_area:
    #     perimeter = cv2.arcLength(con, True)
    #     area = cv2.contourArea(con)
    #     if perimeter == 0:
    #         break

    #     circularity = 4*math.pi*(area/(perimeter*perimeter))
    #     print(circularity)

    #     if 0.7 < circularity < 1.2:
    #         contours_cirles.append(con)

    ### ----------------------------------

    cv2.drawContours(frame, contours, -1, (255, 255, 0), 4)

    if pred is not None:
        x,y,w,h = coords
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        cv2.putText(frame, pred, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)


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

    if args.pic:
        frame = cv2.imread(args.filepath)
        if frame is None:
            print('Failed to open file')
            exit(1)

        process_naming(frame)

        cv2.imshow('frame',frame)
        cv2.waitKey(0)
        exit(1)
    else:
        if args.cam:
            cam_src = '/dev/v4l/by-id/usb-046d_0825_CA00E440-video-index0'  # Logitech camera
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

        while(cap.isOpened()):
            start_time = time.time()

            ret, frame = cap.read()
            if frame is None:
                exit(1)

            frame = cv2.resize(frame, (320, 240))
            process_naming(frame)
            
            end_time = time.time()

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            full_time   += end_time - start_time
            frame_cntr  += 1

            if full_time > measure_time:
                fps = float(frame_cntr) / full_time
                full_time, frame_cntr = 0, 0

                print('FPS: %g' % fps)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    execute_model()
