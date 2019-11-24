import os
import cv2


video = cv2.VideoCapture(0)

n=0
while True:
    ret, frame = video.read()
    cv2.imshow("",frame)
    if cv2.waitKey(50) == 27:
        break
    cv2.imwrite("E:\\project\\data_test3\\image%05d.png"%n, frame)
    n += 1  
video.release()
cv2.destroyAllWindows()