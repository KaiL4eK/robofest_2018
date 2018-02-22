#!/usr/bin/env python2
from __future__ import print_function

import roslib
roslib.load_manifest('robofest_jtsn')

import sys
import rospy
import cv2

# print(cv2.__version__)

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)24/1 ! \
                        nvvidconv flip-method=6 ! video/x-raw, format=(string)I420 ! \
                        videoconvert ! video/x-raw, format=(string)BGR ! \
                        appsink")

if not cap.isOpened():
    print('Cam object is not opened')

rospy.init_node('jtsn_cam', anonymous=True)

videoPub = rospy.Publisher('image', Image, queue_size=10)

while cap.isOpened():
    meta, frame = cap.read()

    frame_gaus = cv2.GaussianBlur(frame, gaussian_blur_ksize, gaussian_blur_sigmaX)

    # frame_gray = cv2.cvtColor(frame_gaus, cv2.COLOR_BGR2GRAY)

    # frame_edges = cv2.Canny(frame_gray, threshold1, threshold2)

    # I want to publish the Canny Edge Image and the original Image
    msg_frame = CvBridge().cv2_to_imgmsg(frame)
    # msg_frame_edges = CvBridge().cv2_to_imgmsg(frame_edges)

    videoPub.publish(msg_frame, "bgr8")
