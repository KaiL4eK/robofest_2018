# http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html

import os
import cv2
import matplotlib.pyplot as plt
import math as m
import numpy as np
from skimage import feature, color, data, exposure

from sklearn import neighbors
from sklearn import svm

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix

import random

import itertools

import pickle

import ml_utils

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

main_path = 'data/signs_only'
neg_path = 'data/negative'

data = []
labels = []

time_measure_sum   = 0
time_measure_times = 0


sign_names = os.listdir(main_path)
for sign_type in sign_names:
	print('Sign type: %s' % sign_type)
	sign_imgs = os.listdir(os.path.join(main_path, sign_type))

	for sign_img in sign_imgs:

		img = cv2.imread(os.path.join(main_path, sign_type, sign_img))

		img_grey = color.rgb2grey(img)
		img_grey = exposure.equalize_hist(img_grey)
		
		fd = ml_utils.get_hog_features(img_grey)

		data.append(fd)
		labels.append(sign_type)

neg_imgs = os.listdir(neg_path)
print('Sign type: negative')
for neg_img in neg_imgs:
	for i in range(3):
		img = cv2.imread(os.path.join(neg_path, neg_img))

		img_height, img_width, channels = img.shape

		portion_size = (64, 64)

		x1 = random.randint(0, img_height-portion_size[0]-1)
		y1 = random.randint(0, img_height-portion_size[1]-1)

		x2, y2 = x1+portion_size[0], y1+portion_size[1]

		part = img[y1:y2, x1:x2]

		img_grey = color.rgb2grey(part)
		img_grey = exposure.equalize_hist(img_grey)

		fd = ml_utils.get_hog_features(img_grey)

		data.append(fd)
		labels.append('negative')

print(np.unique(labels))

# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# model = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')

# http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
model = svm.LinearSVC(C=1)
model.fit(data, labels)

# save the model
filename = 'model.pickle'
s = pickle.dumps(model, pickle.dump(model, open(filename, 'wb'), protocol=2))

# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
kf = KFold(n_splits=3, shuffle=True)
for train_index, test_index in kf.split(data):
	# print("\nTRAIN:", train_index, "\nTEST:", test_index)
	X_train, X_test = [data[i] for i in train_index],   [data[i] for i in test_index]
	y_train, y_test = [labels[i] for i in train_index], [labels[i] for i in test_index]

	model = svm.LinearSVC(C=1)
	model.fit( X_train, y_train )
	print( model.score( X_test, y_test ) )

	y_true = []
	y_pred = []

	for test_fd, test_label in zip( X_test, y_test ):
		pred = model.predict(test_fd.reshape(1, -1))[0]

		y_true.append(test_label)
		y_pred.append(pred.title().lower()) 

	# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
	CM = confusion_matrix(y_true, y_pred)
	print(CM)

	# plt.figure()
	# plot_confusion_matrix(CM, classes=labels, normalize=True,
 #                      		title='Normalized confusion matrix')
	# plt.show()

	# 	img = test_case[0]
	# 	label = test_case[1]

	# 	img = color.rgb2grey(img)

	# 	start = time.time()
	# 	fd = feature.hog(img, orientations=8, pixels_per_cell=(ppc, ppc),
	# 					 cells_per_block=(2, 2), transform_sqrt=True)
	# 	end = time.time()
	# 	time_measure_sum += end - start
	# 	time_measure_times += 1

	# 	pred = model.predict(fd.reshape(1, -1))[0]

	# 	print(pred.title())
	# 	if label != pred.title().lower():
	# 		print('Achtung!!!')

	# 	cv2.imshow('1', img)
	# 	cv2.waitKey(0)


print( 'Hog mean time: %fs' % ml_utils.get_hog_mean_time() )

exit(0)


 

print( fd.size )
print( fd.size / 9 )
print( m.sqrt(fd.size) )

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 255))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()
