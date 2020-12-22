import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

width = 2560     # 가로
height = 1440     # 세로
# width = 640     # 가로
# height = 480     # 세로

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
print(width, height)
# width = 700
# height = 500

fps = 20
fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X') #DIVX 코덱 적용, 코덱 종류 DIVX, XVID, MJPG, X264, WMV1, WMV2
out = cv2.VideoWriter('./DIVX.avi', fcc, fps, (int(width), int(height)))

while True: 
    ret, frame = cap.read()
    # frame = frame.resize(width, height)
    # print(frame)
    cv2.imshow('divx', frame) # 촬영되는 영상보여준다. 
    out.write(frame) # 촬영되는 영상을 저장하는 객체에 써준다.
    
    k = cv2.waitKey(1) & 0xff 
    if k == 27: 
        break
