import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import imageio


# Image read as grayscale
left_img_origin = Image.open('./source_images/tsukuba-l.png')
img_L = left_img_origin.convert('L')  # grayscale
img_L = np.asarray(img_L)

right_img_origin = Image.open('./source_images/tsukuba-r.png')
img_R = right_img_origin.convert('L')  # grayscale
img_R = np.asarray(img_R)


# Image zero padding
img_L = np.pad(img_L, ((3, 3), (3, 3)), 'constant', constant_values=0)
img_R = np.pad(img_R, ((3, 3), (3, 3)), 'constant', constant_values=0)


# Height and Width
H, W = img_L.shape


# Patch size
patch_size = 7
half_size = patch_size // 2


# Use Square dissimilarity
square = True


# Declare depth map
depth = np.zeros(img_L.shape)

for i in range(half_size, H - half_size):
    print(".", end="", flush=True)  # let the user know that computation is on going

    for j in range(half_size, W - half_size):
        mask_R = img_R[i-half_size:i+half_size + 1, j-half_size:j+half_size + 1]  # Patch from left image
        save_small_value = 100000000
        save_j = 0
        for k in range(j, j+64):
            if k < W - 3:
                mask_L = img_L[i-half_size:i+half_size + 1, k-half_size:k+half_size + 1]  # Patch from right image

                # Dissimilarity  function
                if square:
                    difference = np.sum((mask_R - mask_L) ** 2)  # Square dissimilarity
                else:
                    difference = np.sum(abs(mask_R - mask_L))  # Absolute dissimilarity

                if difference < save_small_value:
                    save_small_value = difference
                    save_j = k

        depth[i, j] = abs(save_j - j)


# save image
save_path = './sample_images'
os.makedirs(save_path, exist_ok=True)
if square:
    imageio.imwrite(os.path.join(save_path, 'sample_depth_sqr.png'), depth)
else:
    imageio.imwrite(os.path.join(save_path, 'sample_depth_abs.png'), depth)

# plt.imshow(depth, cmap='gray'); plt.axis('off')
# plt.savefig('baseline_depth.png')

# plt.subplot(131); plt.imshow(np.asarray(left_img_origin)); plt.axis('off')
# plt.subplot(132); plt.imshow(np.asarray(right_img_origin)); plt.axis('off')
# plt.subplot(133); plt.imshow(depth, cmap='gray'); plt.axis('off')
# plt.show()

