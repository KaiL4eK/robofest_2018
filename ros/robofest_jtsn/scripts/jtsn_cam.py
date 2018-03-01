#!/usr/bin/env python2
from __future__ import print_function

import cv2
print(cv2.__version__)

import roslib
roslib.load_manifest('robofest_jtsn')

import sys
import rospy

import numpy as np

# from sensor_msgs.msgImg import Image
# from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import UInt16

cam_id = 'nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)24/1 ! \
          nvvidconv flip-method=6 ! video/x-raw, format=(string)I420 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'
# cam_id = 1

cap = cv2.VideoCapture(cam_id)

if not cap.isOpened():
    print('Cam object is not opened')

rospy.init_node('jtsn_cam')

videoPub = rospy.Publisher('/jtsn_cam/image_raw/compressed', CompressedImage, queue_size=1)
videoFpsPub = rospy.Publisher('/jtsn_cam/image_fps', UInt16, queue_size=1)

msgImg = CompressedImage()
rate = rospy.Rate(20)
size = None

def show_fps(fps):
    msgFps = UInt16()
    msgFps.data = int(fps)

    videoFpsPub.publish(msgFps)
    print('FPS: %g' % fps)

from utils.fps import *
fps = FPS(cb_time=1, cb_func=show_fps)

while cap.isOpened() and not rospy.is_shutdown():
    
    fps.start()

    meta, frame = cap.read()

    if frame is None:
        print('Failed')

    if size is None:
        size = frame.shape
        print(size)

    frame = cv2.flip(frame, 0)

    # frame_gaus = cv2.GaussianBlur(frame, 3, gaussian_blur_sigmaX)

    # frame_gray = cv2.cvtColor(frame_gaus, cv2.COLOR_BGR2GRAY)

    # frame_edges = cv2.Canny(frame_gray, threshold1, threshold2)

    # I want to publish the Canny Edge Image and the original Image
    # msg_frame_edges = CvBridge().cv2_to_imgmsg(frame_edges)

    fps.stop()

    msgImg.format = "jpeg"
    msgImg.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()

    videoPub.publish(msgImg)

    # msg_frame = CvBridge().cv2_to_imgmsg(frame)
    # videoPub.publish(msg_frame, "bgr8")

    # rate.sleep()
