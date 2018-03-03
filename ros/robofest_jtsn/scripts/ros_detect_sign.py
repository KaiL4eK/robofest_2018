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

from utils.fps import *

import rospy
rospy.init_node('sign_detector', anonymous=True)

import rospkg
rospack = rospkg.RosPack()

from sign_naming import *
model_loc = get_network_model()
model_loc_path = os.path.join(rospack.get_path('robofest_jtsn'), 'scripts/sign_localize/weights_best.h5')
model_loc.load_weights(model_loc_path)
model_det_path = os.path.join(rospack.get_path('robofest_jtsn'), 'scripts/sign_detect/model.pickle')
model_det = pickle.load(open(model_det_path, 'rb'))


detector = SignDetector((model_loc, model_det))


from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

sign_pub = rospy.Publisher("sign_name", String, queue_size=10)
img_pub = rospy.Publisher("result_image/compressed", CompressedImage, queue_size=10)
# img_pub = rospy.Publisher("result_image", Image, queue_size=1)

bridge = CvBridge()
msgImg = CompressedImage()
msgImg.format = "jpeg"

class ImageProcessor:
    def __init__(self):
        rospy.Subscriber("sign_image", Image, self.callback, queue_size=1)
        self.cv_image = None

    def callback(self, data):
        try:
            self.cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)


def print_fps(fps):
    pass
    # print('FPS: {}'.format(fps))

if __name__ == '__main__':

    ic = ImageProcessor()
    fps = FPS(2, print_fps)

    while not rospy.is_shutdown():
        fps.start()

        frame = ic.cv_image
        if frame is None:
            continue

        frame = cv2.resize(frame, (320, 240))
        
        sign_name = detector.process_naming(frame)

        fps.stop()

        if sign_name is None:
            sign_name = 'nothing'

        sign_pub.publish(sign_name)
        
        msgImg.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
        img_pub.publish(msgImg)

        # img_pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))
