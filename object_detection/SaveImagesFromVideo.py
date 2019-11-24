import numpy as np 
import cv2

cap = cv2.VideoCapture('video7.avi')
n = 0
while(cap.isOpened()):
    ret, frame = cap.read()

    # img = frame[:, 0: 1440]
    # img = cv2.resize(img, (640,480))
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imwrite("E:\\project\\data_ori\\image%05d.png" %n, frame)
    n += 1
cap.release()
cv2.destroyAllWindows()