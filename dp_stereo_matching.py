import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import imageio
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--left_image', '-l', default = './source_images/Piano-l.png', type=str, help='left image path')
parser.add_argument('--right_image', '-r', default = './source_images/Piano-r.png', type=str, help='right image path')
parser.add_argument('--save_image', '-s', default = './dp_images/dp_cone.png', type=str, help='save image path')
parser.add_argument('--occlusion_cost', '-o', default = 1000, type=int, help='occlusion cost for dp')
parser.add_argument('--block_size', '-p', default = 5, type=int, help='block size for dissimilarity')
parser.add_argument('--camera_baesline', '-c', default = 64, type=int, help='camera baseline, scan line consistency')
parser.add_argument('--match_cost', '-m', default = 'grad', type=str, help='type of cost computation [abs, sqr, ncc, grad]')
parser.add_argument('--filter', '-f', action='store_true') # store_true의 경우 default 값은 false이며, 인자를 적어 주면 true가 저장된다.
parser.add_argument('--filter_type', '-ft', default = 'gaussian', type=str, help='filter type [gaussian, bilateral, guided]')
parser.add_argument('--kernel_size', '-cs', default = 3, type=int, help='gaussian filter kernel size')
parser.add_argument('--filling', '-fl', action='store_true')
parser.add_argument('--fill_type', '-flt', default = 'wmf', type=str, help='occulsion filling type [simple, wmf(=weighted median filter)]')
parser.add_argument('--resize_scale', '-rs', default = 0.3, type=float, help='resize 2014 data set')
args = parser.parse_args()
print(args)


# # read in image
# im = plt.imread('./dp_images/dp_cone_before.png')
# print(im.shape)
# # plot image in color
# plt.imshow(im, cmap="jet")
# #save image in color
# # plt.imsave("color.png", im, cmap="jet")
# plt.show()
# exit(0)

# Image read as grayscale
# left_img_origin = Image.open(args.left_image)
# img_L = left_img_origin.convert('L')  # grayscale

# left_img_origin = np.asarray(left_img_origin)
# img_L = np.asarray(img_L)

# right_img_origin = Image.open(args.right_image)
# img_R = right_img_origin.convert('L')  # grayscale

# right_img_origin = np.asarray(right_img_origin)
# img_R = np.asarray(img_R)

left_img_origin = cv2.imread(args.left_image)
right_img_origin = cv2.imread(args.right_image)
print('image shape:', left_img_origin.shape)

if args.resize_scale != 1:
    left_img_origin = cv2.resize(left_img_origin, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    right_img_origin = cv2.resize(right_img_origin, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    print(args.resize_scale, 'resized image shape:', left_img_origin.shape)

img_L = cv2.cvtColor(left_img_origin, cv2.COLOR_BGR2GRAY)
img_R = cv2.cvtColor(right_img_origin, cv2.COLOR_BGR2GRAY)

# cv2.imshow('left', left_img_origin)
# cv2.imshow('right', right_img_origin)
# cv2.waitKey(0)
# exit(0)

if args.filter:
    print('filter:', args.filter_type)
    if args.filter_type == 'gaussian':
        # Get gaussian filter
        filter = cv2.getGaussianKernel(ksize=args.kernel_size, sigma=1)

        # Make 2D gaussian filter ( Option )
        filter = filter * filter.T

        # Apply filter to image
        img_L = cv2.filter2D(img_L, -1, filter)
        img_R = cv2.filter2D(img_R, -1, filter)

    elif args.filter_type == 'bilateral':
        img_L = cv2.bilateralFilter(img_L, 7, 50, 50)
        img_R = cv2.bilateralFilter(img_R, 7, 50, 50)
        # cv2.imshow('left', img_L)
        # cv2.imshow('right', img_R)
        # cv2.waitKey(0)

    elif args.filter_type == 'guided':
        r = 9
        eps = 0.01
        img_L = cv2.ximgproc.guidedFilter(left_img_origin, img_L, r, eps)
        img_R = cv2.ximgproc.guidedFilter(right_img_origin, img_R, r, eps)
        # cv2.imshow('left', img_L)
        # cv2.imshow('right', img_R)
        # cv2.waitKey(0)


# Height and Width
row_length, column_length, channel_length = left_img_origin.shape

# Patch size
block_size = args.block_size
half_size = block_size // 2

# depth map
disparity_left = np.zeros(img_L.shape)
disparity_right = np.zeros(img_L.shape)

occlusion_cost = args.occlusion_cost
camera_baesline = args.camera_baesline

# matching cost 변화에 따라 파라미터 조정
if args.match_cost == 'ncc':
    occlusion_cost = -0.4
    eps = 1e-6
elif args.match_cost == 'grad':
    # dissimilarity 미리 계산
    occlusion_cost = 40
    threshBorder = 3
    thresColor = 7
    thresGrad = 2
    gamma = 0.11

    # compute gradient from grayscale images
    left_gradient_x = np.gradient(img_L, axis=1)
    left_gradient_x = left_gradient_x + 128 # 음수 값 처리
    left_gradient_y = np.gradient(img_L, axis=0)
    left_gradient_y = left_gradient_y + 128

    right_gradient_x = np.gradient(img_R, axis=1)
    right_gradient_x = right_gradient_x + 128
    right_gradient_y = np.gradient(img_R, axis=0)
    right_gradient_y = right_gradient_y + 128

    cost_volume = np.ones((row_length, column_length, camera_baesline + 1)).astype(np.float32) * threshBorder

    # 0 ~ camera_baesline 에 해당하는 cost 계산
    for d in range(camera_baesline + 1):
        print('grad cost:', d)
        # color cost
        cost_temp = np.ones((row_length, column_length, channel_length)) * threshBorder
        cost_temp[:, d:, :] = right_img_origin[:, :column_length-d, :]
        cost_color = abs(left_img_origin - cost_temp)
        cost_color = np.mean(cost_color, axis = 2) # bgr mean
        cost_color = np.minimum(cost_color, thresColor)

        # gradient cost
        cost_temp = np.ones((row_length, column_length)) * threshBorder
        cost_temp[:, d:] = right_gradient_x[:, :column_length-d]
        cost_grad_x = abs(left_gradient_x - cost_temp)

        cost_temp = np.ones((row_length, column_length)) * threshBorder
        cost_temp[:, d:] = right_gradient_y[:, :column_length-d]
        cost_grad_y = abs(left_gradient_y - cost_temp)

        cost_grad = np.minimum(cost_grad_x + cost_grad_y, thresGrad)
        
        cost_volume[:, :, d] = gamma * cost_color + (1 - gamma) * cost_grad

    # depth_left = np.argmin(cost_volume, axis = 2)
    # imageio.imwrite('./grad_test.png', depth_left)
    # exit(0)


for row_idx in range(half_size, row_length - half_size):
    print('depth [%d / %d]' % (row_idx, row_length - half_size))
    dp = np.zeros((column_length, column_length))
    back_track = np.ones((column_length, column_length))

    for i in range(column_length):
        dp[0, i] = i * occlusion_cost
    for i in range(column_length):
        dp[i, 0] = i * occlusion_cost

    for i in range(half_size, column_length - half_size): # right image
        mask_R = img_R[row_idx-half_size:row_idx+half_size+1, i-half_size:i+half_size+1]
        if args.match_cost == 'grad':
            color_mask_R = right_img_origin[row_idx-half_size:row_idx+half_size+1, i-half_size:i+half_size+1]
        
        for j in range(i, min(i + camera_baesline + 1, column_length - half_size)): # left image - scan line consistency
            mask_L = img_L[row_idx-half_size:row_idx+half_size+1, j-half_size:j+half_size+1]
            if args.match_cost == 'grad':
                color_mask_L = left_img_origin[row_idx-half_size:row_idx+half_size+1, i-half_size:i+half_size+1]

            if args.match_cost == 'sqr':
                dissimilarity = np.sum((mask_R - mask_L) ** 2)
            elif args.match_cost == 'abs':
                dissimilarity = np.sum(abs(mask_R - mask_L))
            elif args.match_cost == 'ncc':
                mask_R = (mask_R - np.mean(mask_R)) / (np.std(mask_R) + eps)
                mask_L = (mask_L - np.mean(mask_L)) / (np.std(mask_L) + eps)
                dissimilarity = -np.sum(mask_R * mask_L) # 음수로 만듦, 값이 작을수록 좋다
            elif args.match_cost == 'grad':
                dissimilarity = np.sum(cost_volume[row_idx-half_size:row_idx+half_size+1, j-half_size:j+half_size+1, j-i])

            # 세방향 값 계산
            min1 = dp[i - 1, j - 1] + dissimilarity
            if j == i + camera_baesline: # left 가 camera baseline 보다 넘어서게 차이날 수는 없음, 대각선 위로 넘어가면 안됨
                min2 = 100000000
            else:
                min2 = dp[i - 1, j] + occlusion_cost
            if i == j: # left 가 right 보다 뒤로 갈 수는 없음, 대각선 아래로 넘어가면 안됨
                min3 = 100000000
            else:    
                min3 = dp[i, j - 1] + occlusion_cost
            cmin = min(min1, min2, min3)
            dp[i, j] = cmin # cost
            if cmin == min1:
                back_track[i, j] = 1
            elif cmin == min2:
                back_track[i, j] = 2
            elif cmin == min3:
                back_track[i, j] = 3

    i = column_length - 1
    j = column_length - 1

    while i != 0 and j != 0:
        assert i <= j and j <= i + camera_baesline
        if back_track[i, j] == 1:
            disparity_right[row_idx, i] = abs(i - j) # disparity
            disparity_left[row_idx, j] = abs(i - j) # disparity
            i = i - 1
            j = j - 1
        elif back_track[i, j] == 2:
            i = i - 1
        elif back_track[i, j] == 3:
            j = j - 1
    # break


disparity_left = disparity_left.astype(np.int8)
disparity_right = disparity_right.astype(np.int8)


if args.filling:
    print('occlusion:', args.fill_type)
    if args.fill_type == 'simple':
        # handle occlusion
        depth_temp = np.zeros(disparity_left.shape)
        for i in range(row_length):
            print('occlusion [%d / %d]' % (i, row_length))
            for j in range(column_length):
                # depth 가 0 이라면 좌우 가까운 값을 가져옴
                if disparity_left[i, j] == 0:
                    to_left = j - 1
                    to_right = j + 1
                    while to_left >= 0 or to_right < column_length:
                        if to_left >= 0 and disparity_left[i, to_left] != 0:
                            depth_temp[i, j] = disparity_left[i, to_left]
                            break
                        if to_right < column_length and disparity_left[i, to_right] != 0:
                            depth_temp[i, j] = disparity_left[i, to_right]
                            break
                        to_left -= 1
                        to_right += 1

        for i in range(row_length):
            print('occlusion copy [%d / %d]' % (i, row_length))
            for j in range(column_length):
                if depth_temp[i, j] != 0:
                    disparity_left[i, j] = depth_temp[i, j]

    elif args.fill_type == 'wmf':
        r_median = 15
        
        depth = disparity_left.copy()
        # left-right consistancy check
        for row in range(row_length):
            for col in range(column_length):
                left_depth = disparity_left[row, col]
                if left_depth > col:
                    continue

                right_depth = disparity_right[row, col-left_depth]
                if abs(left_depth - right_depth) >= 1:
                    depth[row, col] = -1

        occluded_pixel, filled_depth = fill_invalid(depth, camera_baesline)
        # imageio.imwrite('./filled_test.png', filled_depth)
        disparity_left = weighted_median_filter(left_img_origin, filled_depth, occluded_pixel, r_median, camera_baesline)    


# save image
save_path = './dp_images'
os.makedirs(save_path, exist_ok=True)
# imageio.imwrite(os.path.join(save_path, 'dp_depth_%s_%d_%d.png' % (args.match_cost, block_size, occlusion_cost)), disparity_left)
imageio.imwrite(args.save_image, disparity_left)

# plt.subplot(121)
# plt.imshow(disparity_left, cmap='gray'); plt.axis('off')

# plt.subplot(122)
# plt.imshow(disparity_right, cmap='gray'); plt.axis('off')
# plt.show()