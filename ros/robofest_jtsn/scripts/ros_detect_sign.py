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

class ImageProcessor:
    def __init__(self):
        self.sign_pub = rospy.Publisher("sign_name", String, queue_size=1)
        self.img_pub = rospy.Publisher("result_image/compressed", CompressedImage, queue_size=1)

        self.bridge = CvBridge()
        rospy.Subscriber("sign_image", Image, self.callback, queue_size=1)

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        frame = cv2.resize(self.cv_image, (320, 240))
        
        sign_name = detector.process_naming(frame)

        if sign_name is None:
            sign_name = 'nothing'

        self.sign_pub.publish(sign_name)
        
        msgImg = CompressedImage()
        msgImg.format = "jpeg"
        msgImg.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
        self.img_pub.publish(msgImg)

        # self.img_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))


if __name__ == '__main__':

    ic = ImageProcessor()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
