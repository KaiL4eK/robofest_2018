from __future__ import print_function

import sys
sys.path.append('../')

import os
import cv2
import numpy as np
from keras.models import Model
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint, Callback, RemoteMonitor, CSVLogger
from keras.preprocessing.image import ImageDataGenerator

import random
# import itertools

from sklearn.model_selection import train_test_split

from data import *
from net import *

import argparse

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-w', '--weights', action='store', help='Path to weights file')
parser.add_argument('-t', '--test', action='store_true', help='Test model for loss')
parser.add_argument('-l', '--learning_rate', default=1e-3,	action='store', 	 help='Learning rate value')
parser.add_argument('-a', '--augmentation', action='store_true')

args = parser.parse_args()


def preprocess_regress(imgs, masks):
	imgs_p   = np.ndarray((imgs.shape[0],  nn_np_in_size[0],  nn_np_in_size[1], nn_in_chnls), dtype=np.float32)
	masks_p  = np.ndarray((masks.shape[0], nn_np_out_size[0], nn_np_out_size[1], nn_out_chnls), dtype=np.float32)

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
	# model = get_unet(float(args.learning_rate))
	batch_size = 20

	if args.weights:
		model.load_weights(args.weights)

	# monitor = RemoteMonitor()
	csv_logger = CSVLogger('training.log')

	if args.test:
		while True:
			i_image 	= int(random.random() * len(imgs_train))
			img 		= imgs_train[i_image]
			mask 		= imgs_masks_train[i_image]
			
			input_data  = img[np.newaxis,:,:,:]
			output_data = mask[np.newaxis,:,:,:]

			out = model.predict(input_data, verbose=0)
			
			eval_loss = model.evaluate(input_data, output_data)
			print('Eval loss:\t{}\n'.format(eval_loss))

			# img 		= cv2.cvtColor(img, cv2.COLORa_GRAY2BGR)

			mask 		= cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
			mask		= cv2.resize(mask, nn_in_img_size, interpolation = cv2.INTER_NEAREST)

			show_mask   = out[0]
			show_mask 	= cv2.cvtColor(show_mask, cv2.COLOR_GRAY2BGR)
			show_mask 	= cv2.resize(show_mask, nn_in_img_size, interpolation = cv2.INTER_NEAREST)

			cv2.imshow('frame', np.hstack((img, show_mask, mask)))
			key = cv2.waitKey(0)
			if key == ord(' '):
			 	exit(1)		

	else:
		input_data  = imgs_train
		output_data = imgs_masks_train

		if args.augmentation:
			print('-'*30)
			print('Setup data generator...')
			print('-'*30)

			input_data, val_input_data, output_data, val_output_data = \
					train_test_split(input_data, output_data, test_size=0.5, shuffle=True, random_state=42)

			print(input_data.shape, output_data.shape)
			print(val_input_data.shape, val_output_data.shape)

			data_gen_args = dict( width_shift_range=0.1,
								  height_shift_range=0.1,
								  zoom_range=0.1,
								  horizontal_flip=True,
								  fill_mode='constant',
								  cval=0 )

			image_datagen = ImageDataGenerator(**data_gen_args)
			mask_datagen = ImageDataGenerator(**data_gen_args)

			seed = 1
			image_datagen.fit(input_data, augment=True, seed=seed)
			mask_datagen.fit(output_data, augment=True, seed=seed)

			print('-'*30)
			print('Flowing data...')
			print('-'*30)

			# image_generator = image_datagen.flow(input_data, batch_size=batch_size, seed=seed, save_to_dir='flow_dir', save_prefix='img_', save_format='png')
			# mask_generator  = mask_datagen.flow(output_data, batch_size=batch_size, seed=seed, save_to_dir='flow_dir', save_prefix='mask_', save_format='png')

			image_generator = image_datagen.flow(input_data, batch_size=batch_size, seed=seed)
			mask_generator  = mask_datagen.flow(output_data, batch_size=batch_size, seed=seed)

			print('-'*30)
			print('Zipping generators...')
			print('-'*30)

			train_generator = zip(image_generator, mask_generator)

			print('-'*30)
			print('Fitting model...')
			print('-'*30)

			model.fit_generator( train_generator, steps_per_epoch=10, epochs=70000, verbose=2, 
								 validation_data=(val_input_data, val_output_data),
								 callbacks=[ModelCheckpoint('weights_best.h5', monitor='val_loss', 
								 							save_best_only=True, 
								 							save_weights_only=True, verbose=1),
						   					csv_logger])
			print('-'*30)
			print('Fitting model...')
			print('-'*30)
		else:

			model.fit(input_data, output_data, batch_size=batch_size, epochs=70000, verbose=1, shuffle=True, validation_split=0.1,
					  callbacks=[ModelCheckpoint('weights_best.h5', monitor='val_loss', 
								 				  save_best_only=True, 
								 				  save_weights_only=True, verbose=1),
						   		 csv_logger])

if __name__ == '__main__':
	train_regression()
