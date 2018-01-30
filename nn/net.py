from keras.models import Sequential, Model
from keras.losses import binary_crossentropy, mean_squared_error, hinge
from keras.layers import Input, concatenate, Conv2DTranspose, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Dropout, Deconv2D, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils.layer_utils import print_summary
from keras.utils.vis_utils import plot_model
import numpy as np
import cv2

import tensorflow as tf

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

k_decr = 0.8

nn_in_img_size  = (160, 120)
nn_in_chnls		= 3 
nn_out_img_size = (160, 120)
nn_out_chnls	= 1

nn_np_in_size = (nn_in_img_size[1], nn_in_img_size[0])
nn_np_out_size = (nn_out_img_size[1], nn_out_img_size[0])

# class_list = ['no_sign', 'forward', 'right', 'left', 'forward and right', 'brick', 'stop']
# num_classes = len(class_list)

# Output is resized, BGR, mean subtracted, [0, 1.] scaled by values
def preprocess_img(img):
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	img = cv2.resize(img, nn_in_img_size, interpolation = cv2.INTER_LINEAR)
	img = img.astype('float32', copy=False)
	img /= 255.

	if nn_in_chnls == 1:
		return img[:,:,np.newaxis]
	else:
		return img

def preprocess_mask(img):
	img = cv2.resize(img, nn_out_img_size, interpolation = cv2.INTER_NEAREST)
	img = img.astype('float32', copy=False)
	img /= 255.

	if nn_out_chnls == 1:
		return img[:,:,np.newaxis]
	else:
		return img


def get_network_model(lr=1e-3):

	input = Input(shape=(nn_np_in_size[0], nn_np_in_size[1], nn_in_chnls))
	drop0 = Dropout(0.25)(input)

	conv1 = Conv2D(16,(5,5),activation='relu',padding='same', kernel_initializer = 'he_normal')(drop0)
	conv1 = Conv2D(16,(5,5),activation='relu',padding='same', kernel_initializer = 'he_normal')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	drop1 = Dropout(0.25)(pool1)
	
	# norm1 = BatchNormalization()(drop1)

	conv2 = Conv2D(32,(3,3),activation='relu',padding='same', kernel_initializer = 'he_normal')(drop1)
	conv2 = Conv2D(32,(3,3),activation='relu',padding='same', kernel_initializer = 'he_normal')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	drop2 = Dropout(0.25)(pool2)

	conv3 = Conv2D(64,(3,3),activation='relu',padding='same', kernel_initializer = 'he_normal')(drop2)

	up7 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3), conv2], axis=3)
	conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)

	up8 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7), conv1], axis=3)
	conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)

	# norm2 = BatchNormalization()(drop2)

	# conv3 = Conv2D(64,(3,3),activation='relu',padding='same', kernel_initializer = 'he_normal')(drop2)
	# conv3 = Conv2D(64,(3,3),activation='relu',padding='same', kernel_initializer = 'he_normal')(conv3)
	# drop3 = Dropout(0.5)(conv3)

	out   = Conv2D(nn_out_chnls,(1,1),activation='sigmoid',padding='same', kernel_initializer = 'he_normal')(conv8)

	model = Model(input, out)
	
	# optimizer = SGD(lr=lr, momentum=0.9, decay=1e-4)
	optimizer = Adam(lr=lr)#, decay=1e-4)
	
	model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[])

	print_summary(model)
	plot_model(model, show_shapes=True)

	return model

# def get_unet(lr=1e-3):
#     inputs = Input((nn_np_in_size[0], nn_np_in_size[1], 1))
#     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
#     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

#     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
#     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

#     conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
#     conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

#     conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
#     conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

#     up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
#     conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
#     conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

#     up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
#     conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
#     conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

#     up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
#     conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
#     conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

#     up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
#     conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
#     conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

#     conv10 = Conv2D(nn_out_chnls, (1, 1), activation='sigmoid')(conv9)

#     model = Model(inputs=[inputs], outputs=[conv10])

#     model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[])
    
#     print_summary(model)
#     plot_model(model, show_shapes=True)

#     return model


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

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
