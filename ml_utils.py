from skimage import feature, transform
import time

__time_measure_sum   = 0
__time_measure_times = 0

def get_hog_features(img):

	global __time_measure_sum, __time_measure_times

	# configuration
	img_shape = (64, 64)
	ppc = 8

	start = time.time()

	img_height, img_width = img.shape

	if img_height != img_shape[1] or img_width != img_shape[0]:
		img = transform.resize( img, img_shape )

	fd = feature.hog(img, orientations=8, pixels_per_cell=(ppc, ppc),
	            	 cells_per_block=(2, 2), transform_sqrt=True)

	end = time.time()
	__time_measure_sum 	 += end - start
	__time_measure_times += 1

	return fd

def get_hog_mean_time():
	return (__time_measure_sum / __time_measure_times)

