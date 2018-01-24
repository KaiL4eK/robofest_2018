# http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html

import os
import cv2
import matplotlib.pyplot as plt
import math as m
import numpy as np
from skimage import feature, color, data, exposure

from sklearn.neighbors import KNeighborsClassifier

from random import randrange
import time

main_path = 'data/signs_only'

data = []
labels = []

tests = []

time_measure_sum   = 0
time_measure_times = 0

ppc = 8

sign_names = os.listdir(main_path)
for sign_type in sign_names:
	print('Sign type: %s' % sign_type)
	sign_imgs = os.listdir(os.path.join(main_path, sign_type))

	idx = 0

	for sign_img in sign_imgs:

		img = cv2.imread(os.path.join(main_path, sign_type, sign_img))

		img = color.rgb2grey(img)

		start = time.time()
		fd = feature.hog(img, orientations=8, pixels_per_cell=(ppc, ppc),
                    	 cells_per_block=(2, 2), transform_sqrt=True)
		end = time.time()
		time_measure_sum += end - start
		time_measure_times += 1

		# fd, hog_image = feature.hog(img, orientations=8, pixels_per_cell=(8, 8),
                    				# cells_per_block=(2, 2), visualise=True, transform_sqrt=True)

		# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 255))
		# hog_image_rescaled = hog_image_rescaled.astype('uint8')

		# cv2.imshow('1', np.hstack( (img, hog_image) ))
		# cv2.imshow('2', hog_image)
		# cv2.waitKey(10)

		if idx == randrange(0, len(sign_imgs)):
			tests.append( (img, sign_type) )
		else:
			data.append(fd)
			labels.append(sign_type)

		idx += 1

model = KNeighborsClassifier(n_neighbors=3, weights='distance')
model.fit(data, labels)

for test_case in tests:
	img = test_case[0]
	label = test_case[1]

	img = color.rgb2grey(img)

	start = time.time()
	fd = feature.hog(img, orientations=8, pixels_per_cell=(ppc, ppc),
					 cells_per_block=(2, 2), transform_sqrt=True)
	end = time.time()
	time_measure_sum += end - start
	time_measure_times += 1

	pred = model.predict(fd.reshape(1, -1))[0]

	print(pred.title())
	if label != pred.title().lower():
		print('Achtung!!!')

	cv2.imshow('1', img)
	cv2.waitKey(0)


print( time_measure_sum / time_measure_times, ' s' )

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
