from __future__ import print_function

# Enable CPU only
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from imutils.video import FPS
import os
import cv2
import numpy as np
import time
from keras.models import Model, load_model, save_model

import rects as R
from net import *
import argparse

# from skvideo.io import VideoCapture

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('weights', action='store', help='Path to weights file')
parser.add_argument('filepath', action='store', help='Path to video file to process')
parser.add_argument('-f', '--fps', action='store_true', help='Check fps')
parser.add_argument('-p', '--pic', action='store_true', help='Process picture')

args = parser.parse_args()

data_path = 'raw/'

try:
    xrange
except NameError:
    xrange = range

def process_naming(frame, model):
    input = preprocess_img(frame)
    mask = model.predict(np.array([input]))[0] * 255.

    mask = cv2.resize(mask.astype('uint8'), (frame.shape[1], frame.shape[0]))
    # print(mask.shape, input.shape)

    im, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)

    cv2.drawContours(frame, contours, -1, (255, 255, 0), 4)


def execute_model():
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

        cap = cv2.VideoCapture(args.filepath)
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

        if args.fps:
            fps = FPS()
            fps.start()

            bbox_obtain_time = 0
            num_frames = 100
            start = time.time()
            for i in xrange(0, num_frames) :
                ret, frame = cap.read()
                if frame is None:
                    exit(1)

                frame = cv2.resize(frame, (320, 240))

                # start_bbox = time.time()

                process_naming(frame, model)
                
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                fps.update()

            fps.stop()

            print('Estimated frames per second: {0}'.format(fps.fps()))
            print('Elapsed time: {0} ms'.format(fps.elapsed() * 1000))

        else:
            while(cap.isOpened()):
                ret, frame = cap.read()
                if frame is None:
                    exit(1)

                frame = cv2.resize(frame, (320, 240))

                process_naming(frame, model)
                
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    execute_model()
