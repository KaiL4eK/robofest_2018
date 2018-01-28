import cv2

print(cv2.ocl.haveOpenCL())
cv2.ocl.setUseOpenCL(True)
print(cv2.ocl.useOpenCL())

import imutils
import time
import pickle
import math
import numpy as np

from skimage import feature, transform

from skimage import color
from skimage import exposure

val_Hmin = 91
val_Hmax = 127
val_Smin = 164 #75
val_Smax = 255
val_Vmin = 40
val_Vmax = 190

blue_min = (val_Hmin, val_Smin, val_Vmin)
blue_max = (val_Hmax, val_Smax, val_Vmax)

val_Hmin = 0
val_Hmax = 21
val_Smin = 155
val_Smax = 255
val_Vmin = 122
val_Vmax = 190

red_min = (val_Hmin, val_Smin, val_Vmin)
red_max = (val_Hmax, val_Smax, val_Vmax)

import ml_utils

try: 
    xrange 
except NameError: 
    xrange = range

import argparse

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('filepath', action='store', help='Path to video file to process')
# parser.add_argument('reference', action='store', help='Path to reference file')

args = parser.parse_args()

# ref_img = cv2.imread(args.reference)

filename = 'model.pickle'
model = pickle.load(open(filename, 'rb'))

cap = cv2.VideoCapture(args.filepath)
if cap is None or not cap.isOpened():
    print('Failed to open file')
    exit(1)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))

while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        exit(1)

    # sum = np.sum(frame, axis=2) / 3
    # print(frame[:, :, 0].shape)

    work_frm = frame.copy()
    work_frm=cv2.GaussianBlur(work_frm, (5, 5), 0)

    # work_frm = cv2.resize(work_frm, (320, 240))
    work_frm = cv2.resize(work_frm, (640, 480))
    

    work_frm_hsv = cv2.cvtColor(work_frm, cv2.COLOR_BGR2HSV)
    work_frm_gray = cv2.cvtColor(work_frm, cv2.COLOR_BGR2GRAY)
    hsv_filt_blue_frm = cv2.inRange(work_frm_hsv, blue_min, blue_max)

    # --- Erode, dilate -----------

    # eksize = 3
    # morph_frm = cv2.erode(hsv_filt_blue_frm, 
                        # kernel=np.ones((eksize,eksize),np.uint8), 
                        # iterations=2)

    # dksize = 5
    # morph_frm = cv2.dilate(morph_frm, 
                        # kernel=np.ones((dksize,dksize),np.uint8), 
                        # iterations=2)

    oksize = 3
    open_kernel = np.ones((oksize, oksize), np.uint8)
    morph_frm = cv2.morphologyEx(hsv_filt_blue_frm, cv2.MORPH_OPEN, 
                                 kernel=open_kernel, iterations=2) # erosion followed by dilation


    # cksize = 5
    # close_kernel = np.ones((cksize, cksize), np.uint8)
    # morph_frm = cv2.morphologyEx(morph_frm, cv2.MORPH_CLOSE, 
    #                              kernel=close_kernel, iterations=5) # dilation followed by erosion

    # hough_frm = work_frm.copy()

    # canny = cv2.Canny(morph_frm, 100, 200)

    # ------------ Hough ------------
    # circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1,20,
    #                             param1=50,param2=30,minRadius=0,maxRadius=0)
    # if circles is not None:
    #     circles = np.round(circles[0, :]).astype("int")
    #     for (x, y, r) in circles:
    #         cv2.circle(hough_frm, (x, y), r, (0, 0, 255), 4)
    #         # cv2.rectangle(res_frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    # # ------------------------------

    # ------------ Contour -----------
    # im, contours, hierarchy = cv2.findContours(morph_frm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # zones = []
    # contours_area = []
    # contours_cirles = []

    # up_limit    = 20000
    # low_limit   = 300

    # centers = []
    # # calculate area and filter into new array
    # for con in contours:
    #     area = cv2.contourArea(con)
    #     if low_limit < area < up_limit:
    #         contours_area.append(con)
    #         # x,y,w,h = cv2.boundingRect(con)
    #         # centers.append( (x + w/2, y + h/2) )

    # blank_image = np.zeros(hsv_filt_blue_frm.shape, np.uint8)

    # cv2.drawContours(work_frm, contours_area, -1, (255, 255, 0), 4)


    # for cnt in contours_area:
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     cv2.rectangle(work_frm, (x,y), (x+w,y+h), (255, 0, 0), 2)
    #     cv2.rectangle(blank_image, (x,y), (x+w,y+h), 255, -1)


    # dksize = 5
    # blank_image = cv2.dilate(blank_image, 
    #                     kernel=np.ones((dksize,dksize),np.uint8), 
    #                     iterations=4)

    # im, contours, hierarchy = cv2.findContours(blank_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # blank_image = cv2.cvtColor(blank_image, cv2.COLOR_GRAY2BGR)

    # for con in contours:
    #     area = cv2.contourArea(con)
    #     if low_limit < area < up_limit:
    #         contours_area.append(con)
    #         x,y,w,h = cv2.boundingRect(con)
    #         cv2.rectangle(work_frm, (x,y), (x+w,y+h), (255, 0, 255), 2)

    # ---------- Classification -----------------------
            # window_img = work_frm[y:y+h, x:x+w]
            # window_img = color.rgb2grey(window_img)
            # window_img = exposure.equalize_hist(window_img)

            # hf = ml_utils.get_hog_features(window_img)

            # y_pred = model.predict(hf.reshape(1, -1))[0]

            # if y_pred.title().lower() != 'negative':
            #     cv2.putText(work_frm, y_pred.title().lower(), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, 
            #                 (255,0,255), 2, cv2.LINE_AA)


    # -----------------------------------------------

            # centers.append( (x + w/2, y + h/2) )

    # for con in contours_area:
    #     perimeter = cv2.arcLength(con, True)
    #     area = cv2.contourArea(con)
    #     if perimeter == 0:
    #         break

    #     circularity = 4*math.pi*(area/(perimeter*perimeter))
    #     print(circularity)

    #     if 0.7 < circularity < 1.2:
    #         contours_cirles.append(con)

    
    # cv2.drawContours( work_frm, contours_cirles, -1, (255, 0, 0), 4 )

    # --------- HOG ----------------

    # ppc = 8
    # fd = feature.hog(work_frm_gray, orientations=8, pixels_per_cell=(ppc, ppc),
    #                  cells_per_block=(2, 2), transform_sqrt=True)
    # print(work_frm_gray.shape, len(fd))

    # 36192
    # 30x40 = 1200

    # ------------------------------

    # --------------------

    # sum = np.clip(np.sum(work_frm, axis=2) / 3, 1, 255)
    # norm_frame = work_frm[:, :, 0] / sum * 255
    # norm_frame = norm_frame.astype('uint8')
    
    # print(hsv_filt_blue_frm)

    # -------- SURF ------------
    # surf_create_start = time.time()
    # sift = cv2.xfeatures2d.SIFT_create()
    # bf = cv2.BFMatcher()
    # surf_create_end = time.time()
    # kp1, des1 = sift.detectAndCompute(cv2.cvtColor(work_frm, cv2.COLOR_BGR2GRAY), None)
    # kp2, des2 = sift.detectAndCompute(cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY), None)
    # surf_compute_end = time.time()

    # matches = bf.knnMatch(des1, des2, k=2)

    # # Apply ratio test
    # good = []
    # for m,n in matches:
    #     if m.distance < 0.75*n.distance:
    #         good.append([m])

    # # cv2.drawMatchesKnn expects list of lists as matches.
    # img3 = cv2.drawMatchesKnn(work_frm, kp1, ref_img, kp2, good, None, flags=2)

    # Initiate STAR detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(work_frm_gray, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(work_frm_gray, kp)

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(work_frm, kp, None, color=(0,255,0), flags=0)

    # plt.imshow(img3)
    # plt.show()

    # cv2.drawKeypoints(work_frm, kp, work_frm)
    # print('%d, %.2g, %.4f' % (len(kp), (surf_create_end-surf_create_start), (surf_compute_end-surf_create_end)))

    # --------------------------

    # cv2.imshow('1', img3)
    cv2.imshow('2', np.hstack((work_frm, 
                               # blank_image,
                               img2,
                               # cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR),
                               cv2.cvtColor(morph_frm, cv2.COLOR_GRAY2BGR))))
    
    # out.write(work_frm)
    cv2.waitKey(1)

# out.release()

