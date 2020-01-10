import os
import cv2


video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) #3 1280
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) #4 720
#video.set(cv2.CAP_PROP_EXPOSURE, -5) 
#video.set(cv2.CAP_PROP_FPS, 30) 
flag = 1
n=0
while True:
    ret, frame = video.read()
    print(frame.shape)
    cv2.imshow("",frame)

    cv2.imwrite("E:\\project\\test\\image%04d.png"%n, frame)
    # if cv2.waitKey(30) == ord("c"):
    #     cv2.imwrite("E:\\project\\data\\image%0d6.png"%n, frame)
    if cv2.waitKey(100) == 27: 
        break
    # if cv2.waitKey(50) == ord("c"):
    #     flag = 1
    # elif cv2.waitKey(50) == ord("s"):
    #     flag = 0

    # if flag == 0:
    #     continue
    # else: 
    #     cv2.imwrite("E:\\project\\data_non_helmet\\image%06d_nonhelmet.png"%n, frame)
    n += 1  

video.release()
cv2.destroyAllWindows()