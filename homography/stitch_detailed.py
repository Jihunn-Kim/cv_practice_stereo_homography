# https://github.com/opencv/opencv/blob/master/samples/python/stitching_detailed.py
import argparse
import imutils
from imutils.video import WebcamVideoStream
# from webcamvideostream import WebcamVideoStream
import time
import cv2
import numpy as np

WARP_CHOICES = (
    'spherical',
    'plane',
    'affine',
    'cylindrical',
    'fisheye',
    'stereographic',
    'compressedPlaneA2B1',
    'compressedPlaneA1.5B1',
    'compressedPlanePortraitA2B1',
    'compressedPlanePortraitA1.5B1',
    'paniniA2B1',
    'paniniA1.5B1',
    'paniniPortraitA2B1',
    'paniniPortraitA1.5B1',
    'mercator',
    'transverseMercator',
)

WAVE_CORRECT_CHOICES = ('horiz', 'no', 'vert',)

parser = argparse.ArgumentParser(
    prog="stitching_detailed.py", description="Rotation model images stitcher"
)

parser.add_argument(
    '--work_megapix', action='store', default=1.0,
    help="Resolution for image registration step. The default is 0.6 Mpx",
    type=float, dest='work_megapix'
)
parser.add_argument(
    '--ba_refine_mask', action='store', default='xxxxx',
    help="Set refinement mask for bundle adjustment. It looks like 'x_xxx', "
         "where 'x' means refine respective parameter and '_' means don't refine, "
         "and has the following format:<fx><skew><ppx><aspect><ppy>. "
         "The default mask is 'xxxxx'. "
         "If bundle adjustment doesn't support estimation of selected parameter then "
         "the respective flag is ignored.",
    type=str, dest='ba_refine_mask'
)
parser.add_argument(
    '--wave_correct', action='store', default=WAVE_CORRECT_CHOICES[0],
    help="Perform wave effect correction. The default is '%s'" % WAVE_CORRECT_CHOICES[0],
    choices=WAVE_CORRECT_CHOICES,
    type=str, dest='wave_correct'
)
parser.add_argument(
    '--warp', action='store', default=WARP_CHOICES[0],
    help="Warp surface type. The default is '%s'." % WARP_CHOICES[0],
    choices=WARP_CHOICES,
    type=str, dest='warp'
)
parser.add_argument(
    '--seam_megapix', action='store', default=0.75,
    help="Resolution for seam estimation step. The default is 0.1 Mpx.",
    type=float, dest='seam_megapix'
)
parser.add_argument(
    '--compose_megapix', action='store', default=-1,
    help="Resolution for compositing step. Use -1 for original resolution. The default is -1",
    type=float, dest='compose_megapix'
)
parser.add_argument(
    '--blend_strength', action='store', default=5,
    help="Blending strength from [0,100] range. The default is 5",
    type=np.int32, dest='blend_strength'
)
parser.add_argument(
    '--output', action='store', default='result.jpg',
    help="The default is 'result.jpg'",
    type=str, dest='output'
)
parser.add_argument(
    '--num_camera', action='store', default=3,
    help="number of web camera",
    type=int, dest='num_camera'
)


if __name__ == '__main__':
    print("[INFO] starting web cameras...")
    args = parser.parse_args()
    web_cameras = []
    num_camera = args.num_camera
    # width = 2560 # 가로
    # height = 1440 # 세로
    # width = 1200 # 가로
    # height = 800 # 세로
    for num in range(num_camera):
        # web_cameras.append(WebcamVideoStream(width, height, src=num, name='cam%s' % (num)))
        web_cameras.append(WebcamVideoStream(src=num, name='cam%s' % (num)))
        web_cameras[num].start()
        # web_cameras.append(WebcamVideoStream(src=num, name='cam%s' % (num)).start())
        # web_cameras.append(cv2.VideoCapture(num, cv2.CAP_DSHOW))
        # print(web_cameras[num].isOpened())
    # web_cameras.reverse() # order: left -> right
    time.sleep(2.0)

    print("[INFO] initializing parameter")
    work_megapix = args.work_megapix
    seam_megapix = args.seam_megapix
    compose_megapix = args.compose_megapix
    ba_refine_mask = args.ba_refine_mask
    wave_correct = args.wave_correct
    if wave_correct == 'no':
        do_wave_correct = False
    else:
        do_wave_correct = True

    warp_type = args.warp
    blend_strength = args.blend_strength
    result_name = args.output
    seam_work_aspect = 1

    # match images
    # feature_finder = cv2.ORB.create(nfeatures=8000, scaleFactor=1.01)
    # feature_finder = cv2.ORB.create(nfeatures=8000, scaleFactor=1.1)
    # feature_finder = cv2.ORB.create(nfeatures=20000, scaleFactor=1.1)
    # feature_finder = cv2.ORB.create(scaleFactor=1.01)
    feature_finder = cv2.ORB.create()
    features = [None for num in range(num_camera)]
    images = [None for num in range(num_camera)]
    full_img_sizes = [None for num in range(num_camera)]

    #test_img = ["1Hill.jpg", "2Hill.jpg"]

    is_work_scale_set = False
    is_seam_scale_set = False
    for idx in range(num_camera):
        full_img = web_cameras[idx].read()
        # full_img = cv2.imread(test_img[idx])
        cv2.imshow("%s" % (idx), full_img)

        full_img_sizes[idx] = ((full_img.shape[1], full_img.shape[0]))
        if work_megapix < 0:
            img = full_img
            work_scale = 1
            is_work_scale_set = True
        else:
            if is_work_scale_set is False:
                work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                is_work_scale_set = True
            img = cv2.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv2.INTER_LINEAR_EXACT)
        if is_seam_scale_set is False:
            seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            seam_work_aspect = seam_scale / work_scale
            is_seam_scale_set = True
        img_feat = cv2.detail.computeImageFeatures2(feature_finder, img)
        features[idx] = img_feat
        img = cv2.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv2.INTER_LINEAR_EXACT)
        images[idx] = (img)
    # test
    cv2.waitKey()
    #
    matcher = cv2.detail.BestOf2NearestMatcher_create(match_conf=0.3)
    p = matcher.apply2(features)
    matcher.collectGarbage()

    indices = cv2.detail.leaveBiggestComponent(features, p, 0.3)

    img_subset = []
    full_img_sizes_subset = []
    for i in range(len(indices)):
        img_subset.append(images[indices[i, 0]])
        full_img_sizes_subset.append(full_img_sizes[indices[i, 0]])
    images = img_subset
    full_img_sizes = full_img_sizes_subset
    num_images = len(images)
    if num_images < 2:
        print("Need more images")
        exit()
    # end feature, match

    # estimate camera params
    print("[INFO] estimating camera params...")
    estimator = cv2.detail_HomographyBasedEstimator()
    b, cameras = estimator.apply(features, p, None)
    if not b:
        print("Homography estimation failed.")
        exit()
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)

    adjuster = cv2.detail_BundleAdjusterRay()
    adjuster.setConfThresh(1)
    refine_mask = np.zeros((3, 3), np.uint8)
    if ba_refine_mask[0] == 'x':
        refine_mask[0, 0] = 1
    if ba_refine_mask[1] == 'x':
        refine_mask[0, 1] = 1
    if ba_refine_mask[2] == 'x':
        refine_mask[0, 2] = 1
    if ba_refine_mask[3] == 'x':
        refine_mask[1, 1] = 1
    if ba_refine_mask[4] == 'x':
        refine_mask[1, 2] = 1
    adjuster.setRefinementMask(refine_mask)
    b, cameras = adjuster.apply(features, p, cameras)
    if not b:
        print("Camera parameters adjusting failed.")
        exit()
    focals = []
    for cam in cameras:
        focals.append(cam.focal)
    sorted(focals)
    if len(focals) % 2 == 1:
        warped_image_scale = focals[len(focals) // 2]
    else:
        warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
    if do_wave_correct:
        rmats = []
        for cam in cameras:
            rmats.append(np.copy(cam.R))
        rmats = cv2.detail.waveCorrect(rmats, cv2.detail.WAVE_CORRECT_HORIZ)
        for idx, cam in enumerate(cameras):
            cam.R = rmats[idx]
    # end estimate

    # find seam, prepare masks
    print("[INFO] masks..")
    blender = cv2.detail_MultiBandBlender()
    compensator = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS)

    seam_finder = cv2.detail_GraphCutSeamFinder('COST_COLOR')
    corners = []
    images_warped = []
    masks = []
    masks_warped = []
    for i in range(0, num_images):
        um = cv2.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8))
        masks.append(um)

    warper = cv2.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)
    for idx in range(0, num_images):
        K = cameras[idx].K().astype(np.float32)
        swa = seam_work_aspect
        K[0, 0] *= swa
        K[0, 2] *= swa
        K[1, 1] *= swa
        K[1, 2] *= swa
        corner, image_wp = warper.warp(images[idx], K, cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        corners.append(corner)
        images_warped.append(image_wp)

        p, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        masks_warped.append(mask_wp.get())

    compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

    images_warped_f = []
    for img in images_warped:
        imgf = img.astype(np.float32)
        images_warped_f.append(imgf)

    seam_finder.find(images_warped_f, corners, masks_warped)
    # end masks

    # cache crop size
    original_images = [None for num in range(num_camera)]
    result = None
    result_mask = None
    compose_scale = 1

    for idx in range(num_camera):
        original_images[idx] = web_cameras[idx].read()
        
    original_images_subset = []
    for i in range(len(indices)):
        original_images_subset.append(original_images[indices[i, 0]])
    original_images = original_images_subset

    corners = []
    sizes = []
    is_compose_scale_set = False
    is_blender_prepared = False

    for idx, full_img in enumerate(original_images):
        if not is_compose_scale_set:
            if compose_megapix > 0:
                compose_scale = min(1.0, np.sqrt(compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            is_compose_scale_set = True
            compose_work_aspect = compose_scale / work_scale
            warped_image_scale *= compose_work_aspect
            warper = cv2.PyRotationWarper(warp_type, warped_image_scale)
            for i in range(0, len(original_images)):
                cameras[i].focal *= compose_work_aspect
                cameras[i].ppx *= compose_work_aspect
                cameras[i].ppy *= compose_work_aspect
                sz = (full_img_sizes[i][0] * compose_scale, full_img_sizes[i][1] * compose_scale)
                K = cameras[i].K().astype(np.float32)
                roi = warper.warpRoi(sz, K, cameras[i].R)
                corners.append(roi[0:2])
                sizes.append(roi[2:4])
        if abs(compose_scale - 1) > 1e-1:
            img = cv2.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale,
                            interpolation=cv2.INTER_LINEAR_EXACT)
        else:
            img = full_img
        K = cameras[idx].K().astype(np.float32)
        corner, image_warped = warper.warp(img, K, cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
        p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        compensator.apply(idx, corners[idx], image_warped, mask_warped)
        image_warped_s = image_warped.astype(np.int16)
        dilated_mask = cv2.dilate(masks_warped[idx], None)
        seam_mask = cv2.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv2.INTER_LINEAR_EXACT)
        mask_warped = cv2.bitwise_and(seam_mask, mask_warped)

        if not is_blender_prepared:
            dst_sz = cv2.detail.resultRoi(corners=corners, sizes=sizes)
            blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
            blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int))
            blender.prepare(dst_sz)
            is_blender_prepared = True

        blender.feed(cv2.UMat(image_warped_s), mask_warped, corners[idx])

    result, result_mask = blender.blend(result, result_mask)
    stitched = cv2.normalize(src=result, dst=None, alpha=255., norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # cv2.imshow("before crop", stitched)

    print("[INFO] cropping...")
    stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,
        cv2.BORDER_CONSTANT, (0, 0, 0))

    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('tgr', gray)
    smallest = 0
    # smallest = np.amin(gray)
    thresh = cv2.threshold(gray, smallest, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow('ther', thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    minRect = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)
    
    cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
    # end crop size

    # compose panorama
    print("[INFO] composing...")
    original_images = [None for num in range(num_camera)]
    result = None
    result_mask = None
    for idx in range(num_camera):
        masks_warped[idx] = cv2.dilate(masks_warped[idx], None)
    
    fps = 10
    fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X') #DIVX 코덱 적용, 코덱 종류 DIVX, XVID, MJPG, X264, WMV1, WMV2
    width = stitched.shape[1]
    height = stitched.shape[0]
    print(width, height)
    out = cv2.VideoWriter('./DIVX_10.avi', fcc, fps, (width, height))

    before_ = time.clock()
    while True:
        for idx in range(num_camera):
            original_images[idx] = web_cameras[idx].read()
            # cv2.imshow("%s" % (idx + 1), images[idx])
            
        original_images_subset = []
        for i in range(len(indices)):
            original_images_subset.append(original_images[indices[i, 0]])
        original_images = original_images_subset
        
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        
        blender.prepare(dst_sz)
        for idx, full_img in enumerate(original_images):
            if abs(compose_scale - 1) > 1e-1:
                img = cv2.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale,
                                interpolation=cv2.INTER_LINEAR_EXACT)
            else:
                img = full_img
            K = cameras[idx].K().astype(np.float32)
            corner, image_warped = warper.warp(img, K, cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
            compensator.apply(idx, corners[idx], image_warped, mask_warped)
            image_warped_s = image_warped.astype(np.int16)
            seam_mask = cv2.resize(masks_warped[idx], (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv2.INTER_LINEAR_EXACT)
            mask_warped = cv2.bitwise_and(seam_mask, mask_warped)
            
            blender.feed(cv2.UMat(image_warped_s), mask_warped, corners[idx])

        result, result_mask = blender.blend(result, result_mask)
        # zoom_x = 600.0 / result.shape[1]
        # zoom_x = 0.80
        dst = cv2.normalize(src=result, dst=None, alpha=255., norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dst = dst[y + 20:y + h - 20, x:x + w]
        cv2.imshow(result_name, dst)
        
        # dst = cv2.resize(dst, dsize=(width, height))
        # dst = cv.resize(dst, dsize=None, fx=zoom_x, fy=zoom_x)
        # out.write(dst)

    # print("[INFO] fps: %.2f" % (total_frame / (time.time() - start)))
    print(time.clock() - before_)
    # end panorama

    print("[INFO] cleaning up...")
    for num in range(num_camera):
        web_cameras[num].stop()
        # web_cameras[num].release()
    out.release()
    cv2.destroyAllWindows()
    print("done")
    exit(0)