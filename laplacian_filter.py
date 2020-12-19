import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
filename = './source_images/cameraman.jpg'
img = plt.imread(filename)


# Make laplacian filter
filter = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])


# Apply filter to image
dst = cv2.filter2D(img, -1, filter)


# Plot images
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst, cmap='gray'), plt.title('Laplacian_filter')
plt.xticks([]), plt.yticks([])
plt.show()