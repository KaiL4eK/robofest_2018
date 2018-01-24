# http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html

import matplotlib.pyplot as plt
import math as m
from skimage import feature, color, data, exposure


image = data.astronaut()

image = color.rgb2grey(image)

fd, hog_image = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualise=True, transform_sqrt=True)
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
