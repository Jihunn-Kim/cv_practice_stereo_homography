import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import imageio
import os
import argparse
import time
import numpy.matlib
from scipy.signal import medfilt2d
from scipy.signal import medfilt
from scipy.sparse import coo_matrix
from scipy.sparse import bsr_matrix
import math


###
# Fast Cost-Volume Filtering for Visual Correspondence and Beyond,
# Christoph Rhemann, Asmaa Hosni, Michael Bleyer, Carsten Rother, Margrit Gelautz, CVPR 2011
# 이해를 위해 수정해본 코드
###

save_path = './fast_images'
os.makedirs(save_path, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument('--left_image', '-l', default = './source_images/tsukuba-l.png', type=str, help='left image path')
parser.add_argument('--right_image', '-r', default = './source_images/tsukuba-r.png', type=str, help='right image path')
parser.add_argument('--disparity_image', '-o', default = './fast_images/fast_tsukuba_my2.png', type=str, help='disparity image path')
parser.add_argument('--bilateral', action='store_true', help='use bilateral filter for cost aggregation, default=guided filter')
args = parser.parse_args()


# guided filter constants
r = 9
eps = 0.01 # 0.0001 # 엡실론이 작을수록 엣지가 살아나며 클수록 mean필터에 가까워짐


# cost matching constants
thresColor = 7./255.
thresGrad = 2./255.
threshBorder = 3./255.
gamma = 0.11

# weight median filter constants
sigma_c = 0.1
sigma_s = 9
r_median = 19 # window size


def computeDisp(left_image_path, right_image_path, max_disp):
    # Image read as grayscale
    left_img_origin = Image.open(left_image_path)
    img_L = left_img_origin.convert('L')  # grayscale

    left_img_origin = np.asarray(left_img_origin).astype(np.float32) / 255.
    img_L = np.asarray(img_L).astype(np.float32) / 255.

    right_img_origin = Image.open(right_image_path)
    img_R = right_img_origin.convert('L')  # grayscale

    right_img_origin = np.asarray(right_img_origin).astype(np.float32) / 255.
    img_R = np.asarray(img_R).astype(np.float32) / 255.

    # Height and Width
    row_length, column_length, channel_length = left_img_origin.shape

    # compute matching cost
    tic = time.time()

    # compute gradient from grayscale images
    left_gradient_x = np.gradient(img_L, axis=1)
    left_gradient_x = left_gradient_x + 0.5 # 음수 값 처리 [-0.5 ~ 0.5] -> [0 ~ 1.0]
    left_gradient_y = np.gradient(img_L, axis=0)
    left_gradient_y = left_gradient_y + 0.5

    right_gradient_x = np.gradient(img_R, axis=1)
    right_gradient_x = right_gradient_x + 0.5
    right_gradient_y = np.gradient(img_R, axis=0)
    right_gradient_y = right_gradient_y + 0.5

    cost_volume_left = np.ones((row_length, column_length, max_disp)).astype(np.float32) * threshBorder
    cost_volume_right = np.ones((row_length, column_length, max_disp)).astype(np.float32) * threshBorder
    # end

    # 1 ~ max_disp 에 해당하는 cost 계산
    for d in range(1, max_disp+1):
        # left cost volume
        # color cost
        cost_temp = np.ones((row_length, column_length, channel_length)) * threshBorder
        cost_temp[:, d:, :] = right_img_origin[:, :column_length-d, :]
        cost_color_left = abs(left_img_origin - cost_temp)
        cost_color_left = np.mean(cost_color_left, axis = 2) # bgr mean
        cost_color_left = np.minimum(cost_color_left, thresColor)

        # gradient cost
        cost_temp = np.ones((row_length, column_length)) * threshBorder
        cost_temp[:, d:] = right_gradient_x[:, :column_length-d]
        cost_grad_left_x = abs(left_gradient_x - cost_temp)

        cost_temp = np.ones((row_length, column_length)) * threshBorder
        cost_temp[:, d:] = right_gradient_y[:, :column_length-d]
        cost_grad_left_y = abs(left_gradient_y - cost_temp)

        cost_grad_left = np.minimum(cost_grad_left_x + cost_grad_left_y, thresGrad)
        
        cost_volume_left[:, :, d-1] = gamma * cost_color_left + (1 - gamma) * cost_grad_left
        # end left

        # right cost volume
        # color cost
        cost_temp = np.ones((row_length, column_length, channel_length)) * threshBorder
        cost_temp[:, :column_length-d, :] = left_img_origin[:, d:, :]
        cost_color_right = abs(right_img_origin - cost_temp)
        cost_color_right = np.mean(cost_color_right, axis = 2) # bgr mean
        cost_color_right = np.minimum(cost_color_right, thresColor)

        # gradient cost
        cost_temp = np.ones((row_length, column_length)) * threshBorder
        cost_temp[:, :column_length-d] = left_gradient_x[:, d:]
        cost_grad_right_x = abs(right_gradient_x - cost_temp)

        cost_temp = np.ones((row_length, column_length)) * threshBorder
        cost_temp[:, :column_length-d] = left_gradient_y[:, d:]
        cost_grad_right_y = abs(right_gradient_y - cost_temp)

        cost_grad_right = np.minimum(cost_grad_right_x + cost_grad_right_y, thresGrad)

        cost_volume_right[:, :, d-1] = gamma * cost_color_right + (1 - gamma) * cost_grad_right
        # end right

    toc = time.time()
    print('* Elapsed time (cost computation): %f sec.' % (toc - tic))


    # cost aggregation, guidedFilter or bilateralFilter
    tic = time.time()
    for d in range(max_disp):
        if args.bilateral:
            cost_volume_left[:, :, d] = cv2.bilateralFilter(cost_volume_left[:, :, d], 13, 90, 90)
            cost_volume_right[:, :, d] = cv2.bilateralFilter(cost_volume_right[:, :, d], 13, 90, 90)
        else:
            cost_volume_left[:, :, d] = cv2.ximgproc.guidedFilter(left_img_origin, cost_volume_left[:, :, d], r, eps)
            cost_volume_right[:, :, d] = cv2.ximgproc.guidedFilter(right_img_origin, cost_volume_right[:, :, d], r, eps)
    
    toc = time.time()
    print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))


    # disparity computation and optimization, using winner-take-all
    tic = time.time()

    depth_left = np.argmin(cost_volume_left, axis = 2)
    depth_right = np.argmin(cost_volume_right, axis = 2)
    # imageio.imwrite('./left_test.png', depth_left)
    # imageio.imwrite('./right_test.png', depth_right)

    toc = time.time()
    print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))


    # Disparity refinement, left-right consistancy check 후 잘못 된 픽셀에 weighted median filter
    tic = time.time()

    depth = depth_left.copy()
    # left-right consistancy check
    for row in range(row_length):
        for col in range(column_length):
            left_depth = depth_left[row, col]
            if left_depth > col:
                continue
            right_depth = depth_right[row, col-left_depth]

            if abs(left_depth - right_depth) >= 1:
                depth[row, col] = -1

    # imageio.imwrite('./check_test.png', depth)

    occluded_pixel, filled_depth = fill_invalid(depth, max_disp)
    # imageio.imwrite('./filled_test.png', filled_depth)

    final_depth = weighted_median_filter(left_img_origin, filled_depth, occluded_pixel, r_median, max_disp)    

    imageio.imwrite(args.disparity_image, final_depth)
    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))


def fill_invalid(depth, max_disp):
    row_length, column_length = depth.shape
    # occlude 위치 저장
    occluded_pixel = np.zeros((row_length, column_length))
    occluded_pixel[depth<0] = 1
    
    # left
    fillVals = np.ones((row_length)) * max_disp
    depth_left = depth.copy()
    for col in range(column_length):
        curCol = depth[:,col].copy()
        curCol[curCol==-1] = fillVals[curCol==-1] # 현재 열에서 occluded 된 위치 가져옴
        fillVals[curCol!=-1] = curCol[curCol!=-1] # 다음 열로 전달
        depth_left[:,col] = curCol
    
    # right
    fillVals = np.ones((row_length)) * max_disp
    depth_right = depth.copy()
    for col in reversed(range(column_length)):
        curCol = depth[:,col].copy()
        curCol[curCol==-1] = fillVals[curCol==-1]
        fillVals[curCol!=-1] = curCol[curCol!=-1]
        depth_right[:,col] = curCol

    filled_depth = np.fmin(depth_left, depth_right)
    return occluded_pixel, filled_depth


def weighted_median_filter(left_img_origin, filled_depth, occluded_pixel, r_median, max_disp):
    row_length, column_length, _ = left_img_origin.shape
    filtered_img = cv2.medianBlur(left_img_origin, 3) # 이미지 median blur

    window_size = r_median // 2
    weighted_median = np.ones((row_length, column_length)) * -1

    for row in range(row_length):
        print('[%d/%d]' % (row, row_length))
        for col in range(column_length):
            if occluded_pixel[row, col] == 0: # occluede 아닌 픽셀은 무시함
                continue

            weights = [0.0 for _ in range(max_disp+1)]
            total_sum = 0.0
            for i in range(max(row - window_size, 0), min(row + window_size, row_length)):
                for j in range(max(col - window_size, 0), min(col + window_size, column_length)):
                    # 현재 픽셀에서 멀리 떨어질수록 weight 적게 줌
                    spatial_diff = np.sqrt( np.square(row-i) + np.square(col-j) )
                    # color 값 변화가 크면 weight 를 적게줌
                    color_diff = np.sqrt( np.sum( np.square(filtered_img[row, col, :] - filtered_img[i, j, :]), axis=-1 )  )

                    weight = np.exp( - spatial_diff/(sigma_s*sigma_s) - color_diff/(sigma_c*sigma_c) )
                    weights[filled_depth[i, j]] += weight
                    total_sum += weight
            
            # 임계값을 넘어간 순간 occlude 픽셀을 해당 값으로 치환
            cum_sum = 0.0
            for i in range(max_disp+1):
                cum_sum += weights[i]
                if (cum_sum > total_sum / 2):
                    weighted_median[row, col] = i
                    break
    
    filled_depth[weighted_median!=-1] = weighted_median[weighted_median!=-1]
    return filled_depth


def main():
    # 카메라 베이스라인 최대치를 16 으로
    max_disp = 16
    computeDisp(args.left_image, args.right_image, max_disp)


if __name__ == '__main__':
    main()