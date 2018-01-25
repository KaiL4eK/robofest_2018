
import cv2
import imutils
import time
import pickle

import numpy as np

from skimage import color
from skimage import exposure

import ml_utils

try: 
	xrange 
except NameError: 
	xrange = range

import argparse

parser = argparse.ArgumentParser(description='Process video with ANN')
parser.add_argument('filepath', action='store', help='Path to video file to process')

args = parser.parse_args()

cap = cv2.VideoCapture(args.filepath)
if cap is None or not cap.isOpened():
	print('Failed to open file')
	exit(1)

ret, frame = cap.read()
if frame is None:
	print('Failed to read frame')
	exit(1)

# https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

filename = 'model.pickle'
model = pickle.load(open(filename, 'rb'))

while cap.isOpened():
	ret, frame = cap.read()
	if frame is None:
		exit(1)

	(winW, winH) = (120, 120)

	# for resized in pyramid(frame, scale=2, minSize=(100, 100)):
		# print(resized.shape)
		# loop over the sliding window for each layer of the pyramid

	work_img = frame.copy()

	work_img = cv2.GaussianBlur(work_img,(5,5),0)

	gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
	# gray = exposure.equalize_hist(gray)
	# gray = cv2.equalizeHist(gray)
	# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	# gray = clahe.apply(gray)

	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5, 100)

	if circles is not None:
		# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")
	 
		# loop over the (x, y) coordinates and radius of the circles
		for (x, y, r) in circles:
			# draw the circle in the output image, then draw a rectangle
			# corresponding to the center of the circle
			cv2.circle(work_img, (x, y), r, (0, 255, 0), 4)
			cv2.rectangle(work_img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
	 
		# show the output image
		cv2.imshow("output", np.hstack([frame, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), work_img]))
		cv2.waitKey(1)

	continue

	for (x, y, window) in sliding_window(resized, stepSize=20, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
 
		# clone = resized.copy()
		# cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)

		window_img = resized[y:y+winH, x:x+winW]
		window_img = color.rgb2grey(window_img)
		window_img = exposure.equalize_hist(window_img)

		hf = ml_utils.get_hog_features(window_img)

		y_pred = model.predict(hf.reshape(1, -1))[0]

		# print(y_pred.title())
		if y_pred.title().lower() != 'negative':
			print(y_pred.title().lower())
			cv2.rectangle(resized, (x, y), (x + winW, y + winH), (0, 0, 255), 2)

		# cv2.imshow("Part", window_img)
	cv2.imshow("Window", resized)
	cv2.waitKey(1)
		# time.sleep(0.02)

		# cv2.waitKey(0)
	# cv2.imshow('frame',frame)



	# wait_res = cv2.waitKey(1)
	# if wait_res & 0xFF == ord('q'):
	# 	break

