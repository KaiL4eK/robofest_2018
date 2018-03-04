#!/usr/bin/env python

import numpy as np
import rospy
from sensor_msgs.msg import LaserScan, PointCloud2


# import bottleneck as bn
import time

# scan - 0:360 degrees (fi)
def process_lines(scan):
    # Rho and Theta ranges
    thetas  = np.deg2rad(np.arange(-180.0, 180.0))
    fis     = np.deg2rad(np.arange(-180.0, 180.0))
    diag_len = 50   # max_dist = 5,0 meters (*10)
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    r_idxs = [i for i in range(len(scan)) if scan[i] > 0.05]
    # print(r_idxs, len(r_idxs))

    # Vote in the hough accumulator
    for r_idx in r_idxs:
        fi  = fis[r_idx]
        rng = scan[r_idx] * 10

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            # rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
            angl = int(fi + thetas[t_idx])
            # if angl >= 360:
                # angl = 0

            rho = diag_len + (rng * sin_t[angl])
            accumulator[int(rho), t_idx] += 1

    return accumulator, thetas, rhos

class HoughLines:
    def __init__(self):
        self.range_limits = [0.05, 5]
        self.angle_limits = []

    def laser_cb(self, msg):
        # print(msg)
        self.ranges = msg.ranges
        # print(np.array(self.ranges).shape)
        time_start = time.time()
        accumulator, thetas, rhos = process_lines(self.ranges)
        print((time.time() - time_start)*1000)

        idx = np.argmax(accumulator)    # returns flattened
        rho_idx = idx / accumulator.shape[1]
        theta_idx = idx % accumulator.shape[1]

        rho = rhos[rho_idx] / 10.
        theta = thetas[theta_idx]

        # print(accumulator[accumulator >= 295])

        # idx = bn.argpartsort(accumulator, accumulator.size-3, axis=None)[-3:]
        # width = accumulator.shape[1]
        # print [divmod(i, width) for i in idx]

        # print(-np.cos(theta)/np.sin(theta), rho/np.sin(theta))

        print "rho={0:.2f}, theta={1:.0f}".format(rho, np.rad2deg(theta))


################################################################################

from geometry_msgs.msg import Point

show_roi = True

class ObstacleAvoider:
    def __init__(self):
        rospy.Subscriber('scan', LaserScan, self.neato_laser_cb, queue_size=1)

        if show_roi:
            self.roi_obst_pub = rospy.Publisher('roi_scan', LaserScan, queue_size=1)

        self.range_size = 20
        self.angle_idxs = np.arange(-self.range_size, self.range_size+1)
        self.angles     = np.deg2rad(self.angle_idxs)
        self.len_angles = len(self.angles)

        self.cos_t = np.cos(self.angles)
        self.sin_t = np.sin(self.angles)

        self.range_min  = 0.06

        self.coordinates = []
        self.obstX       = []
        self.obstY       = []

        for idx in range(self.len_angles):
            self.coordinates.append( Point() )
            self.obstX.append(0)
            self.obstY.append(0)

    def neato_laser_cb(self, neato_msg):
        ranges = np.array(neato_msg.ranges)

        roi_obstacle_scan = ranges[self.angle_idxs]

        for idx, angle in enumerate(self.angles):
            self.coordinates[idx].x = roi_obstacle_scan[idx] * self.cos_t[idx]
            self.coordinates[idx].y = roi_obstacle_scan[idx] * self.sin_t[idx]

            self.obstX[idx] = self.coordinates[idx].x
            self.obstY[idx] = self.coordinates[idx].y


        self.check_obstacle()

        if show_roi:
            msg                 = neato_msg
            msg.angle_min       = np.deg2rad(-self.range_size)
            msg.angle_max       = np.deg2rad(self.range_size)

            msg.ranges          = roi_obstacle_scan

            self.roi_obst_pub.publish(msg)

            # pc_msg              = PointCloud2()
            # pc_msg.header       = neato_msg.header
            # pc_msg.height       = 1
            # pc_msg.width        = len(angle_idxs)

            # for idx, coord in enumerate(coordinates):
            #     pc_msg.points[idx].x = coord[0]
            #     pc_msg.points[idx].y = coord[1]

            # pc_pub.publish(pc_msg)

    def check_obstacle(self):
        for idx, coord in enumerate(self.coordinates):
            if abs(coord.y) < 0.15 and abs(coord.x) > self.range_min and coord.x < 2.5: 
                print(coord.x, coord.y)
     
        print('----------------------')

if __name__ == '__main__':
    hough = HoughLines()

    rospy.init_node('hough_test')
    
    obstAv = ObstacleAvoider()

    rospy.spin()
