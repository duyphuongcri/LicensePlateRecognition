######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Duy-Phuong Dao
# E-mail: duypphuongcri@gmail.com

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb





# Import packages
from __future__ import print_function
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

import time
import argparse
from datetime import datetime

def detect_line(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bw = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)

    lines = cv2.HoughLinesP(img_bw, rho=1, theta=np.pi/180, threshold=150, minLineLength=10, maxLineGap=30)
    #print(lines[0])
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        slope = (y2-y1)/(x2-x1)
        intercept = y1 - slope*x1
        print([slope, intercept])
        if -0.1 < slope < 0.1 and 100 < intercept < 240:
            print(" 0 =  {0:.2f} * x + {1:.2f} - y".format(slope, intercept))
            break
            #cv2.line(img, (x1, y1), ( x2, y2), (0,255,0), 5 )
    return slope, intercept

def visualize_image(image_ori, list_License_Plate_box, vehicle_boxes):
    # All the results have been drawn on image. Now display the image.
    for i, box in enumerate(list_License_Plate_box):
        if len(box) != 0 and list_real_plate_mode[i]:
            cv2.rectangle(image_ori, (box[1], box[0]),(box[3], box[2]), (0,0,255), 2)
            cv2.putText(image_ori, "Plate", (box[1], box[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),3)  

    for i,box in enumerate(vehicle_boxes):
        cv2.rectangle(image_ori, (box[1], box[0]),(box[3], box[2]), (0,0,255), 2)
        cv2.putText(image_ori, str(class_text[classID[i]]), (box[1], box[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),3)

    # Visualize date time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    cv2.putText(image_ori, dt_string, (5, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255),1)

    return image_ori

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

def detect_plate(input_image, list_License_Plate_box):
    list_real_plate_mode = []
    list_number_plates = []
    for i, box in enumerate(list_License_Plate_box):
        list_number = []
        image = input_image[box[0]: box[2], box[1]: box[3]]
        h_plate, w_plate = image.shape[:2]
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean, stv = cv2.meanStdDev(img_gray)
        print("mean, stev", mean, stv)
        if mean > 90 and w_plate < 100 and h_plate < 100 and h_plate < w_plate:
            list_real_plate_mode.append(True)
        else:
            list_real_plate_mode.append(False)
            continue

        fn = filename.replace(".png", "{}.png".format(i))
        cv2.imwrite("")
        
        # Chuyển sang ảnh nhị phân ( ngưỡng = 150); > 150 => 255;   < 150 ==> 0
        # Ảnh nhị phân là ảnh trắng đen ( chỉ có 2 màu trắng và đen tương ứng giá trị 255 và 0)
        #img_gray = cv2.GaussianBlur(img_gray,(5,5),0)
        #_, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) # 

        size_block = 3
        decrease_value = 5 # a constant which is subtracted from the mean or weighted mean calculated
        img_bw = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, size_block, decrease_value)

        # img_bw_resized = cv2.resize(img_bw,(200, int(h_plate*200/w_plate)))
        # h_resized, w_resized = img_bw_resized.shape[:2]


        # Tìm đường biên của các vùng màu trắng (foreground)
        _, contours, _ = cv2.findContours(img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Lấy ra đường biên cùng vùng có diện tích lớn nhất
        contours = sorted(contours, key = cv2.contourArea, reverse=True)
        # # tạo 1 ảnh mới có kích thước bằng với ảnh đầu vào
        mask_top = np.zeros(img_bw.shape[:2], dtype=np.uint8)
        mask_bot = np.zeros(img_bw.shape[:2], dtype=np.uint8)
        mask = np.zeros(img_bw.shape[:2], dtype=np.uint8)
        n = 0
        print(len(contours))
        for c in contours:
            flag = False
            x, y, w, h = cv2.boundingRect(c)
            if n == 9:
                break
            if w_plate / h_plate < 2:
                if h < h_plate / 2  and w < w_plate / 5 and y < h_plate * 0.3 and h > w and h_plate / 2.8 > h > h_plate / 5 :
                    cv2.rectangle(mask,(x, y),(x+w,y+h),(255),1)   
                    cv2.drawContours(mask_top, [c], -1, (255), -1)
                    mask_top = cv2.bitwise_and(img_bw,mask_top)
                    flag = True
                    n += 1
                elif h < h_plate / 2 and w < w_plate / 5  and h_plate * 0.8 > y > h_plate*0.3 and h > w and h_plate / 2.8 > h > h_plate / 5:
                    cv2.rectangle(mask,(x, y),(x+w,y+h),(255),1) 
                    cv2.drawContours(mask_bot, [c], -1, (255), -1)
                    mask_bot = cv2.bitwise_and(img_bw,mask_bot) 
                    flag = True
                    n += 1  
            elif h >= h_plate / 2 and h > w:
                cv2.rectangle(mask,(x, y),(x+w,y+h),(255),1)   
                cv2.drawContours(mask_top, [c], -1, (255), -1)    
                mask_top = cv2.bitwise_and(img_bw,mask_top)
                flag = True        
                n += 1               
            # if flag:
            #     fn = filename.replace(".png", "{}.png".format(i))

        mask_top = cv2.resize(mask_top, (200, int(h_plate*200/w_plate)))
        mask_bot = cv2.resize(mask_bot, (200, int(h_plate*200/w_plate)))

        _, mask_top_bw = cv2.threshold(mask_top, 50, 255, cv2.THRESH_BINARY)
        _, mask_bot_bw = cv2.threshold(mask_bot, 50, 255, cv2.THRESH_BINARY)
        # Tìm đường biên của các vùng màu trắng (foreground)
        _, contours_top, _ = cv2.findContours(mask_top_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        _, contours_bot, _ = cv2.findContours(mask_bot_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Lấy ra đường biên cùng vùng có diện tích lớn nhất
        k = 0
        for c in contours_top:
            x, y, w, h = cv2.boundingRect(c)
            fn = filename.replace(".png", "{}.png".format(k))
            #cv2.imwrite("E:\\project\\letter\\" + fn, mask_top_bw[y:y+h, x:x+w])
            k +=1
        for c in contours_bot:
            x, y, w, h = cv2.boundingRect(c)
            fn = filename.replace(".png", "{}.png".format(k))
            # predict letter
            output = predict_letter(mask_bot_bw[y:y+h, x:x+w], model)
            list_number.append(output)
            #cv2.imwrite("E:\\project\\letter\\" + fn, mask_bot_bw[y:y+h, x:x+w])
            k +=1

        list_number_plates.append(list_number)
        cv2.imshow("mask_top{}".format(i), mask_top_bw)
        cv2.imshow("Mask_bot{}".format(i), mask_bot_bw)
        cv2.imshow("bw{}".format(i), cv2.resize(img_bw, (200, int(h_plate*200/w_plate))))

 

    return list_real_plate_mode, list_number_plates

def crop_plate_region(input_image, vehicle_boxes, classID):
    list_region_plate = []
    list_x1_region = []
    list_mode_crop = []
    for i, box in enumerate(vehicle_boxes):  
        img_region_plate =  input_image[box[0]: box[2], box[1]: box[3]]
        # padding to get quare image
        if box[2] - box[0] > box[3] - box[1]: # h > w 
            right_pad = (box[2] - box[0]) - (box[3] - box[1])
            padding = [(0, 0), (0, right_pad), (0, 0)]
            img_region_plate = np.pad(img_region_plate, padding, mode='constant', constant_values=0)
        elif box[2] - box[0] < box[3] - box[1]: # h < w 
            bottom_pad = (box[3] - box[1]) - (box[2] - box[0])
            padding = [(0, bottom_pad), (0, 0), (0, 0)]
            img_region_plate = np.pad(img_region_plate, padding, mode='constant', constant_values=0)
        
        ###
        if img_region_plate.shape[0] > 300:
            img_region_plate = img_region_plate[0:300, 0:300]

        list_region_plate.append(img_region_plate)   
    return list_region_plate


def detect_License_Plate(list_region_plate, 
                            vehicle_boxes, 
                            sess, 
                            detection_boxes, 
                            detection_scores,
                            detection_classes, 
                            num_detections, 
                            image_tensor, 
                            threshold_score):
    list_License_Plate_box = []
    
    for i, image in enumerate(list_region_plate):
        img_resized = cv2.resize(image,(300,300))
        # print(img_resized.shape)
        # cv2.imshow("", img_resized)
        # cv2.waitKey(0)
        image_expanded = np.expand_dims(img_resized, axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        for j in range(1):
            if scores[0][j] > threshold_score:
                y1, x1, y2, x2 = int(boxes[0][j][0]*img_resized.shape[0]), int(boxes[0][j][1]*img_resized.shape[1]), int(boxes[0][j][2]*img_resized.shape[0]), int(boxes[0][j][3]*img_resized.shape[1])
                ratio = image.shape[0] / img_resized.shape[0] 
                y1_Plate_ori, x1_Plate_ori, y2_Plate_ori, x2_Plate_ori = int(y1*ratio + vehicle_boxes[i][0]), int(x1*ratio + vehicle_boxes[i][1]), int(y2*ratio + + vehicle_boxes[i][0]), int(x2*ratio + + vehicle_boxes[i][1])
      
                list_License_Plate_box.append([y1_Plate_ori, x1_Plate_ori, y2_Plate_ori, x2_Plate_ori])
            else:
                list_License_Plate_box.append([])
 
    return list_License_Plate_box
            
def detect_vehicle(input_image, sess, detection_boxes, detection_scores, detection_classes, num_detections, image_tensor, threshold_score):
    h_ori, w_ori = input_image.shape[:2]

    ##Resize input image
    image = cv2.resize(input_image, (300,300))
    image_expanded = np.expand_dims(image, axis=0)
     
    #image = input_image
    #image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    vehicle_boxes = []
    classID = []
    for i in range(10):
        if scores[0][i] > threshold_score:
            y1, x1, y2, x2 = int(boxes[0][i][0]*image.shape[0]), int(boxes[0][i][1]*image.shape[1]), int(boxes[0][i][2]*image.shape[0]), int(boxes[0][i][3]*image.shape[1])
            # mapping coordinate from resized image to original image
            hor_ratio, ver_ratio = input_image.shape[1] / image.shape[1], input_image.shape[0] / image.shape[0]
            y1_vehicle_ori, x1_vehicle_ori, y2_vehicle_ori, x2_vehicle_ori = int(y1*ver_ratio), int(x1*hor_ratio), int(y2*ver_ratio), int(x2*hor_ratio)
            if x1_vehicle_ori > 10:
                x1_vehicle_ori = x1_vehicle_ori - 10
            else:
                x1_vehicle_ori = 0
            vehicle_boxes.append([y1_vehicle_ori, x1_vehicle_ori, y2_vehicle_ori, x2_vehicle_ori])
            classID.append(int(classes[0][i]))
        else:
            break
    return vehicle_boxes, classID



# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT_vehicle = os.path.join(CWD_PATH,"Vehicle_ssd",'frozen_inference_graph.pb')
PATH_TO_CKPT_LicensePlate = os.path.join(CWD_PATH,"License_PLate",'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS_vehicle = os.path.join(CWD_PATH,'training','labelmap_vehicle.pbtxt')
PATH_TO_LABELS_LicensePlate = os.path.join(CWD_PATH,'training','labelmap_LicensePlate.pbtxt')


# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map_vehicle = label_map_util.load_labelmap(PATH_TO_LABELS_vehicle)
categories_vehicle = label_map_util.convert_label_map_to_categories(label_map_vehicle, max_num_classes=2, use_display_name=True)
category_index_vehicle = label_map_util.create_category_index(categories_vehicle)

label_map_LicensePlate = label_map_util.load_labelmap(PATH_TO_LABELS_LicensePlate)
categories_LicensePlatev = label_map_util.convert_label_map_to_categories(label_map_LicensePlate, max_num_classes=1, use_display_name=True)
category_index_LicensePlate = label_map_util.create_category_index(categories_LicensePlatev)

# Load the Tensorflow model into memory.
detection_graph_vehicle = tf.Graph()
with detection_graph_vehicle.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT_vehicle, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess_vehicle = tf.Session(graph=detection_graph_vehicle)

detection_graph_LicensePlate = tf.Graph()
with detection_graph_LicensePlate.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT_LicensePlate, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess_LicensePlate = tf.Session(graph=detection_graph_LicensePlate)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor_vehicle = detection_graph_vehicle.get_tensor_by_name('image_tensor:0')
image_tensor_LicensePlate = detection_graph_LicensePlate.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes_vehicle = detection_graph_vehicle.get_tensor_by_name('detection_boxes:0')
detection_boxes_LicensePlate = detection_graph_LicensePlate.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores_vehicle = detection_graph_vehicle.get_tensor_by_name('detection_scores:0')
detection_scores_LicensePlate = detection_graph_LicensePlate.get_tensor_by_name('detection_classes:0')

detection_classes_vehicle = detection_graph_vehicle.get_tensor_by_name('detection_classes:0')
detection_classes_LicensePlate  = detection_graph_LicensePlate .get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections_vehicle = detection_graph_vehicle.get_tensor_by_name('num_detections:0')
num_detections_LicensePlate = detection_graph_LicensePlate.get_tensor_by_name('num_detections:0')



#------------------
from keras.models import load_model
def fit_size(img):
    h, w = img.shape[:2]
    print("Shape of Image:", h, w)
    pd = int((h - w)/2)
    padding = [(0, 0), (pd, pd)]
    img_padded = np.pad(img, padding, mode='constant', constant_values=0)
    output = cv2.resize(img_padded, (64,64), interpolation=cv2.INTER_AREA)
    return output
def predict_letter(img, model):
    img = fit_size(img)

    img = img.reshape(1,64,64,1)
    img = img / 255
    img_class = model.predict(img)
    return img_class.argmax()

model = load_model('./logs/0002.h5')

if __name__=="__main__":
    # Parse command line arguments
    # parser = argparse.ArgumentParser(
    #     description='Detect vehicle and License Plate using Object Detection .')
    # # parser.add_argument("command",
    # #                     metavar="<command>",
    # #                     help="'train' or 'test'")
    # parser.add_argument('--data_dir', required=False,
    #                     metavar="/home/simon/mask_rcnn/data/fci",
    #                     help='Directory of the fci dataset')
    # parser.add_argument('--logs', required=False,
    #                     default=DEFAULT_LOGS_DIR,
    #                     metavar="/path/to/logs/",
    #                     help='Logs and checkpoints directory (default=logs/)')
    # parser.add_argument('--weights', required=False,
    #                     metavar="/home/simon/logs/weights.h5",
    #                     help="Path to weights .h5 file or 'coco'")
 
    # parser.add_argument('--jsonconfig_path', default=os.path.join(ROOT_DIR,'mrcnn/sample_config.json'),
    #                     help='path to config json file')
    # args = parser.parse_args()


    path = "E:\\project\\data"
    files = [i for i in os.listdir(path) if i.endswith("image05771.png")]
    class_text = ["Background","motorbike", "car"]
    first_frame = True
    for filename in files:
        start = time.time()
        print(filename)
        image_ori = cv2.imread(os.path.join(path, filename))
        # if first_frame:
        #     slope, intercept = detect_line(image_ori)
        #     first_frame = False
        if image_ori is None:
            continue
        vehicle_boxes, classID = detect_vehicle(image_ori, 
                                                sess_vehicle, 
                                                detection_boxes_vehicle, 
                                                detection_scores_vehicle, 
                                                detection_classes_vehicle, 
                                                num_detections_vehicle, 
                                                image_tensor_vehicle,
                                                threshold_score=0.8)

        # ## Crop region of interest of License Plate to pass another model which is used to detect License Plate 
        #list_region_plate, list_x1_region, list_mode_crop = crop_plate_region(image_ori, vehicle_boxes, classID)
        list_region_plate = crop_plate_region(image_ori, vehicle_boxes, classID)

        # for i, image in enumerate(list_region_plate):
        #     print(image.shape)
        #     image = cv2.resize(image, (300,300))
        #     print(image.shape)
        #     fn = filename.replace(".png", "{}.png".format(i))
        #     cv2.imwrite("E:\\project\\data_LicensePlate\\"+ fn, image)

        list_License_Plate_box = detect_License_Plate(list_region_plate, 
                                                        vehicle_boxes,
                                                        sess_LicensePlate, 
                                                        detection_boxes_LicensePlate, 
                                                        detection_scores_LicensePlate, 
                                                        detection_classes_LicensePlate, 
                                                        num_detections_LicensePlate, 
                                                        image_tensor_LicensePlate,
                                                        threshold_score=0.5)   

        list_real_plate_mode, list_number_plates = detect_plate(image_ori, list_License_Plate_box)
        print(list_number_plates)
        # for i, box in enumerate(list_License_Plate_box):
        #     image = image_ori[box[0]: box[2], box[1]: box[3]]
        #     fn = filename.replace(".png", "{}.png".format(i))
        #     cv2.imwrite("E:\\project\\LicensePlate\\"+ fn, image)

        image_ori = visualize_image(image_ori, list_License_Plate_box, vehicle_boxes)
        cv2.imshow("", image_ori) 
        if cv2.waitKey(0) == 27:
            break
        print("timing: ",time.time() - start)


    cv2.destroyAllWindows()