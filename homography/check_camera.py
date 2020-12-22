from imutils.video import WebcamVideoStream
# from webcamvideostream import WebcamVideoStream
import time
import cv2 as cv

web_cameras = []
num_camera = 3
for num in range(num_camera):
    web_cameras.append(WebcamVideoStream(src=num, name='cam%s' % (num)).start())
# web_cameras.reverse() # order: left -> right
time.sleep(2.0)

original_images = [None for num in range(num_camera)]
while True:
    for idx in range(num_camera):
        original_images[idx] = web_cameras[idx].read()
        cv.imshow("%s" % (idx), original_images[idx])
    
    key = cv.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

print("[INFO] cleaning up...")
for num in range(num_camera):
    web_cameras[num].stop()
cv.destroyAllWindows()
exit(0)