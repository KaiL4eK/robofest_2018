#!/usr/bin/env python2
from __future__ import print_function

import cv2
print(cv2.__version__)

import roslib
roslib.load_manifest('robofest_jtsn')

import sys
import rospy

import numpy as np
import time

# from sensor_msgs.msgImg import Image
# from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import UInt16

cam_id = 'nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)24/1 ! \
          nvvidconv flip-method=6 ! video/x-raw, format=(string)I420 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'
cam_id = '/dev/v4l/by-id/usb-046d_0825_CA00E440-video-index0'

cap = cv2.VideoCapture(cam_id)

if not cap.isOpened():
    print('Cam object is not opened')

rospy.init_node('jtsn_cam')

videoPub = rospy.Publisher('/jtsn_cam/image_raw/compressed', CompressedImage, queue_size=10)
videoFpsPub = rospy.Publisher('/jtsn_cam/image_fps', UInt16, queue_size=10)

msgImg = CompressedImage()
rate = rospy.Rate(20)
size = None

start_time, end_time, full_time = 0, 0, 0
frame_cntr   = 0
measure_time = 0.01

while cap.isOpened() and not rospy.is_shutdown():
    
    start_time = time.time()

    meta, frame = cap.read()

    if size is None:
        size = frame.shape
        print(size)

    # frame_gaus = cv2.GaussianBlur(frame, 3, gaussian_blur_sigmaX)

    # frame_gray = cv2.cvtColor(frame_gaus, cv2.COLOR_BGR2GRAY)

    # frame_edges = cv2.Canny(frame_gray, threshold1, threshold2)

    # I want to publish the Canny Edge Image and the original Image
    # msg_frame_edges = CvBridge().cv2_to_imgmsg(frame_edges)

    end_time = time.time()
    full_time   += end_time - start_time
    frame_cntr  += 1
    # print(full_time)

    if full_time > measure_time:
        fps = float(frame_cntr) / full_time
        full_time  = 0
        frame_cntr = 0

        msgFps = UInt16()
        msgFps.data = int(fps)

        videoFpsPub.publish(msgFps)


    frame = cv2.flip(frame, 0)

    msgImg.header.stamp = rospy.Time.now()
    msgImg.format = "jpeg"
    msgImg.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()

    videoPub.publish(msgImg)

    # msg_frame = CvBridge().cv2_to_imgmsg(frame)
    # videoPub.publish(msg_frame, "bgr8")

    # rate.sleep()
