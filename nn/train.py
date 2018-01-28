from __future__ import print_function

import sys
sys.path.append('../')

import os
import cv2
import numpy as np
from keras.models import Model
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint, Callback, RemoteMonitor, CSVLogger
import random

from data import *
from net import *
import argparse

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-w', '--weights', action='store', help='Path to weights file')
parser.add_argument('-t', '--test', action='store_true', help='Test model for loss')
parser.add_argument('-l', '--learning_rate', default=1e-3,	action='store', 	 help='Learning rate value')

args = parser.parse_args()


def preprocess_regress(imgs, masks):
	imgs_p   = np.ndarray((imgs.shape[0],  nn_np_in_size[0],  nn_np_in_size[1], 3), dtype=np.float32)
	masks_p  = np.ndarray((masks.shape[0], nn_np_out_size[0], nn_np_out_size[1], 1), dtype=np.float32)

	for i in range(imgs.shape[0]):
		imgs_p[i]   = preprocess_img(imgs[i])
		masks_p[i]	= preprocess_mask(masks[i])

	return imgs_p, masks_p


print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)

imgs_train 			= npy_data_load_images()
imgs_masks_train 	= npy_data_load_masks()
imgs_train, imgs_masks_train = preprocess_regress(imgs_train, imgs_masks_train)


def train_regression():
	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)

	model = get_network_model(float(args.learning_rate))

	if args.weights:
		model.load_weights(args.weights)

	monitor = RemoteMonitor()
	csv_logger = CSVLogger('training.log')

	if args.test:
		i_image 	= int(random.random() * len(imgs_train))
		img 		= np.reshape(imgs_train[i_image], 		(1, nn_np_in_size[0],  nn_np_in_size[1], 3))
		mask 		= np.reshape(imgs_masks_train[i_image], (1, nn_np_out_size[0], nn_np_out_size[1], 1))
		
		input_data  = img
		output_data = mask

		out = model.predict(input_data, verbose=0)
		
		eval_loss = model.evaluate(input_data, output_data)
		print('Eval loss:\t{}\n'.format(eval_loss))

		show_img 		= input_data[0]
		show_mask_true 	= output_data[0]
		show_mask_true 	= cv2.cvtColor(show_mask_true, cv2.COLOR_GRAY2BGR)
		show_mask_true	= cv2.resize(show_mask_true, nn_in_img_size, interpolation = cv2.INTER_NEAREST)

		show_mask   	= out[0]
		show_mask 		= cv2.cvtColor(show_mask, cv2.COLOR_GRAY2BGR)
		show_mask 		= cv2.resize(show_mask, nn_in_img_size, interpolation = cv2.INTER_NEAREST)

		cv2.imshow('frame', np.hstack((show_img, show_mask, show_mask_true)))
		cv2.waitKey(0)
	else:
		print('-'*30)
		print('Fitting model...')
		print('-'*30)

		input_data  = imgs_train
		output_data = imgs_masks_train

		model.fit(input_data, output_data, batch_size=20, epochs=10000, verbose=1, shuffle=True, validation_split=0.1,
					callbacks=[ModelCheckpoint('weights_best.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1),
							   monitor, csv_logger])

if __name__ == '__main__':
	train_regression()
