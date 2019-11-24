import os
import cv2
import numpy as np 

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

path = "E:\\project\\data_test2"
files = [i for i in os.listdir(path) if i.endswith(".png")]

for filename in files:
    print(filename)
    # Đọc ảnh
    img = cv2.imread(os.path.join(path, filename))
    #img_blur = cv2.medianBlur(img, 3)
    # Chuyển ảnh bgr sang gray
    # Ảnh gray là ảnh có mức xám ( giá trị chạy từ 0->255)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Chuyển sang ảnh nhị phân ( ngưỡng = 150); > 150 => 255;   < 150 ==> 0
    # Ảnh nhị phân là ảnh trắng đen ( chỉ có 2 màu trắng và đen tương ứng giá trị 255 và 0)
    _, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # # Tìm đường biên của các vùng màu trắng (foreground)
    # _, contours, _ = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    # # Lấy ra đường biên cùng vùng có diện tích lớn nhất
    # contours = sorted(contours, key = cv2.contourArea, reverse=True)[:1]
    # print(contours)
    # # tạo 1 ảnh mới có kích thước bằng với ảnh đầu vào
    # mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # for c in contours:
    #     rect = cv2.minAreaRect(c) # return (x,y), (w, h), angle
    #     print(rect)
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     cv2.drawContours(mask,[box],-1,(255),-1)    

    # # rect = cv2.minAreaRect(contours[0]) # return (x,y), (w, h), angle
    # # box = cv2.boxPoints(rect)
    # # box = np.int0(box)
    # # cv2.drawContours(mask,[box],-1,(255),-1)    
    # # contours[0]


    # mask = cv2.bitwise_and(img_bw, mask)

    # mask = rotate_bound(mask, -(90 + rect[2]))

    # result = np.where(mask == 255)

    # y1, x1 = min(result[0]), min(result[1])
    # y2, x2 = max(result[0]), max(result[1])
    
    # offset = 1
    # img_cropped = img[y1:y2, x1:x2]
    # license_plate = mask[y1 + offset:y2 - offset, x1 + offset:x2 - offset]

    # license_plate = 255 - license_plate

    # # h, w = license_plate.shape[:2]
    # # license_plate = cv2.resize(license_plate, (100, int(h*100/w)))
    # ret, contours, hierarchy = cv2.findContours(license_plate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    # contours = sorted(contours, key = cv2.contourArea, reverse = True)[:9] 
    # for c in contours:
    #     cv2.drawContours(license_plate,[c],-1,(255),1)

    h, w = img_bw.shape[:2]
    img_bw = cv2.resize(img_bw, (100, int(h*100/w)))
    #img_bw = 255 -img_bw

    # kernel = np.ones((3,3), np.uint8) 
    # img_bw = cv2.erode(img_bw, kernel, iterations=1) 

    #cv2.imshow("", cv2.resize(img, (400,300)))
    cv2.imshow("asdsa", img_bw)
    if cv2.waitKey(0) == 27:
        break

cv2.destroyAllWindows()