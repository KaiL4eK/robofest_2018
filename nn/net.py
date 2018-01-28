from keras.models import Sequential, Model
from keras.losses import binary_crossentropy, mean_squared_error, hinge
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Dropout, Deconv2D, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils.layer_utils import print_summary
from keras.utils.vis_utils import plot_model
import numpy as np
import cv2

import tensorflow as tf

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

nn_in_img_size = (320, 240)
nn_out_img_size = (80, 60)

nn_np_in_size = (nn_in_img_size[1], nn_in_img_size[0])
nn_np_out_size = (nn_out_img_size[1], nn_out_img_size[0])

# class_list = ['no_sign', 'forward', 'right', 'left', 'forward and right', 'brick', 'stop']
# num_classes = len(class_list)

# Output is resized, BGR, mean subtracted, [0, 1.] scaled by values
def preprocess_img(img):
	img = cv2.resize(img, nn_in_img_size, interpolation = cv2.INTER_LINEAR)
	img = img.astype('float32', copy=False)
	img /= 255.
	return img

def preprocess_mask(img):
	img = cv2.resize(img, nn_out_img_size, interpolation = cv2.INTER_NEAREST)
	img = img.astype('float32', copy=False)
	img /= 255.
	img = np.reshape(img, (nn_np_out_size[0], nn_np_out_size[1], 1))
	return img

# norm1 = BatchNormalization()(drop1)

def get_network_model(lr=1e-3):

	input = Input(shape=(nn_np_in_size[0], nn_np_in_size[1], 3))
	drop0 = Dropout(0.25)(input)

	conv1 = Conv2D(8,(11,11),activation='relu',padding='same')(drop0)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	drop1 = Dropout(0.25)(pool1)
	
	conv2 = Conv2D(8,(7,7),activation='relu',padding='same')(drop1)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	drop2 = Dropout(0.25)(pool2)

	conv3 = Conv2D(16,(5,5),activation='relu',padding='same')(drop2)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	drop3 = Dropout(0.25)(pool3)

	conv4 = Conv2D(64,(3,3),activation='relu',padding='same')(drop3)

	dconv1= Conv2D(16,(5,5),activation='relu',padding='same')(conv4)
	upool1= UpSampling2D(size=(2, 2))(dconv1)

	drop4 = Dropout(0.5)(upool1)
	out   = Conv2D(1,(3,3),activation='sigmoid',padding='same')(drop4)

	model = Model(input, out)
	
	optimizer = SGD(lr=lr, momentum=0.9, decay=1e-5)
	optimizer = Adam(lr=lr, decay=1e-5)
	
	model.compile(optimizer=optimizer, loss=iou_loss, metrics=[])

	print_summary(model)
	plot_model(model, show_shapes=True)

	return model

# ----- Losses -----

def intersect_over_union(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)

	intersection = K.sum(y_true_f * y_pred_f)
	union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
	# return K.switch( K.equal(union, 0), K.variable(1), intersection / union) 
	return intersection / union 

def iou_loss(y_true, y_pred):
	return 1 - intersect_over_union(y_true, y_pred)
