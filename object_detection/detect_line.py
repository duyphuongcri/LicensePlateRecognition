import os
import cv2
import numpy as np 

path = "E:\\project\\data_test2"
files = [i for i in os.listdir(path) if i.endswith(".png")]

for filename in files:
    print(filename)
    img = cv2.imread(os.path.join(path, filename))
    #img_blur = cv2.medianBlur(img, 3)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bw = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)

    lines = cv2.HoughLinesP(img_bw, rho=1, theta=np.pi/180, threshold=200, minLineLength=10, maxLineGap=30)
    print(lines)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        slope = (y2-y1)/(x2-x1)
        intercept = y1 - slope*x1
        #print([slope, intercept]) # y =ax+b
        if -0.1 < slope < 0.1 and 100 < intercept < 240:
            #print(" 0 =  {0:.2f} * x + {1:.2f} - y".format(slope, intercept))
            cv2.line(img, (0, int(intercept)), (640, int(slope * 640 + intercept)), (0,255,0), 2)
        else:
            continue
    cv2.imshow("", img)
    if cv2.waitKey(0) == 27:
        break

cv2.destroyAllWindows()