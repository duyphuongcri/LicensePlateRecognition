######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Duy-Phuong Dao
# E-mail: duypphuongcri@gmail.com
#############################################################################

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
import serial
import pandas as pd

##################
import pygame
pygame.mixer.init()
def vuot_den_do_audio():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load('E:\\LicensePlateRecognition\\audio\\a.mp3')
        pygame.mixer.music.play()
##################
def non_max_suppression(boxes, scores, threshold):	
    assert boxes.shape[0] == scores.shape[0]
    # bottom-left origin
    ys1 = boxes[:, 0]
    xs1 = boxes[:, 1]
    # top-right target
    ys2 = boxes[:, 2]
    xs2 = boxes[:, 3]
    # box coordinate ranges are inclusive-inclusive
    areas = (ys2 - ys1) * (xs2 - xs1)
    scores_indexes = scores.argsort().tolist()
    boxes_keep_index = []
    while len(scores_indexes):
        index = scores_indexes.pop()
        boxes_keep_index.append(index)
        if not len(scores_indexes):
            break
        ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index],
                           areas[scores_indexes])
        filtered_indexes = set((ious > threshold).nonzero()[0])
        # if there are no more scores_index
        # then we should pop it
        scores_indexes = [
            v for (i, v) in enumerate(scores_indexes)
            if i not in filtered_indexes
        ]
    return np.array(boxes_keep_index)

def compute_iou(box, boxes, box_area, boxes_area):
    # this is the iou of the box against all other boxes
    assert boxes.shape[0] == boxes_area.shape[0]
    # get all the origin-ys
    # push up all the lower origin-xs, while keeping the higher origin-xs
    ys1 = np.maximum(box[0], boxes[:, 0])
    # get all the origin-xs
    # push right all the lower origin-xs, while keeping higher origin-xs
    xs1 = np.maximum(box[1], boxes[:, 1])
    # get all the target-ys
    # pull down all the higher target-ys, while keeping lower origin-ys
    ys2 = np.minimum(box[2], boxes[:, 2])
    # get all the target-xs
    # pull left all the higher target-xs, while keeping lower target-xs
    xs2 = np.minimum(box[3], boxes[:, 3])
    # each intersection area is calculated by the
    # pulled target-x minus the pushed origin-x
    # multiplying
    # pulled target-y minus the pushed origin-y
    # we ignore areas where the intersection side would be negative
    # this is done by using maxing the side length by 0
    intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
    # each union is then the box area
    # added to each other box area minusing their intersection calculated above
    unions = box_area + boxes_area - intersections
    # element wise division
    # if the intersection is 0, then their ratio is 0
    ious = intersections / unions
    return ious

def check_pass_red_light(vehicle_boxes, slope, intercept):
    list_traffic_violation_mode =[]
    for i, box in enumerate(vehicle_boxes):
        if len(box) == 0:
            list_traffic_violation_mode.append(False)
            continue
        h, w = box[2] - box[0], box[3] - box[1]
        coor_x = box[1] + w * 0.8
        coor_y = box[0] + h * 0.6
        if slope * coor_x + intercept - coor_y > 0:
            list_traffic_violation_mode.append(True)
        else:
            list_traffic_violation_mode.append(False)
    return list_traffic_violation_mode

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
        if 0 < slope < 0.1 and 100 < intercept < 240:
            print(" 0 =  {0:.2f} * x + {1:.2f} - y".format(slope, intercept))
            break
            #cv2.line(img, (x1, y1), ( x2, y2), (0,255,0), 5 )

    mask = np.zeros(img_bw.shape, dtype=np.uint8)
    cv2.line(mask, (0, int(slope * 0 + intercept)), (640, int(slope * 640 + intercept)), (255), 4)
    

    return slope, intercept, mask

def visualize_image(image_ori, list_License_Plate_box, vehicle_boxes, list_full_number_plates, flag_red_light, list_traffic_violation_mode, mask_line):
    
    # draw mask of line
    ret, img_bw_150 = cv2.threshold(cv2.cvtColor(image_ori, cv2.COLOR_BGR2GRAY), 120, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask_line, img_bw_150)
    image_ori[:, :, 0] = np.where(mask == 255 ,100, image_ori[:, :, 0])
    image_ori[:, :, 1] = np.where(mask == 255 ,91, image_ori[:, :, 1])
    image_ori[:, :, 2] = np.where(mask == 255 ,210, image_ori[:, :, 2])
 
    # # draw vehicles
    for i,box in enumerate(vehicle_boxes):
        if list_traffic_violation_mode[i]:
            cv2.rectangle(image_ori, (box[1], box[0]),(box[3], box[2]), (0,0,255), 2)
        else:
            cv2.rectangle(image_ori, (box[1], box[0]),(box[3], box[2]), (0,255,0), 2)
        cv2.putText(image_ori, str(class_text[classID[i]]), (box[1], box[2] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),2)
        if list_traffic_violation_mode[i] and flag_red_light:
            image_ori[box[0]: box[2], box[1]: box[3], 2] = 250

    # # # # # Draw license plate
    for i, box in enumerate(list_License_Plate_box):
        if len(box) == 0:
            continue
        if list_real_plate_mode[i]:
            cv2.putText(image_ori, "Plate", (box[1], box[0] - 5 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)  
            cv2.rectangle(image_ori, (box[1], box[0]),(box[3], box[2]), (0,255,0), 2)   
        if len(list_full_number_plates) > 0:
            cv2.putText(image_ori, list_full_number_plates[0] , (box[1], box[2]+ 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 50, 50),2) 

    # # Visualize date time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    cv2.putText(image_ori, dt_string, (5, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),2)

    # display location
    cv2.putText(image_ori, """Location: 10*14'09.9"N 106*22'20.6"E""", (5, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),2) 

    # display traffic light mode
    cv2.putText(image_ori, "Traffic Light Status: ", (5, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),2) 
    if flag_red_light:
        cv2.circle(image_ori, (210,435), 13, (0,0,255), -1)
        # pass red light or not
        if True in list_traffic_violation_mode:
            cv2.putText(image_ori, "Traffic Violation", (450, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),2) 
    else:
        cv2.circle(image_ori, (210,435), 13, (0,255,0), -1)
    

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

def recognize_plate(input_image, 
                    list_License_Plate_box,
                    sess, 
                    detection_boxes, 
                    detection_scores,
                    detection_classes, 
                    num_detections, 
                    image_tensor, 
                    threshold_score):
    #######################################
    # list_real_plate_mode : False mean this box is plate, 
    #
    #
    list_real_plate_mode = []
    list_number_plates = []
    for i, box in enumerate(list_License_Plate_box):
        if len(box) == 0:
            list_real_plate_mode.append(False)
            list_number_plates.append(False)
            continue
        list_number = []
        image = input_image[box[0]: box[2], box[1]: box[3]]
        h_plate, w_plate = image.shape[:2]
        print("Shape of plate box: ",h_plate, w_plate)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean, stv = cv2.meanStdDev(img_gray)
        if mean > 90 and w_plate < 100 and h_plate < 100 and h_plate < w_plate:
            list_real_plate_mode.append(True)
        else:
            list_real_plate_mode.append(False)
            continue

        pd = w_plate - h_plate
        if pd > 0:
            padding = [(0, pd), (0, 0),(0,0)]
        else:
            padding = [(0, 0), (0, -pd),(0,0)]
        image_padding = np.pad(image, padding, mode='constant', constant_values=0)
        image_resized = cv2.resize(image_padding,(300, 300), interpolation=cv2.INTER_AREA)
        image_expanded = np.expand_dims(image_resized, axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        n_box = 9
        boxes[0][:,0] = boxes[0][:,0]*image_resized.shape[0]
        boxes[0][:,1] = boxes[0][:,1]*image_resized.shape[1]
        boxes[0][:,2] = boxes[0][:,2]*image_resized.shape[0]
        boxes[0][:,3] = boxes[0][:,3]*image_resized.shape[1]
        # perform non-maximum suppression on the bounding boxes
        index = non_max_suppression(boxes[0][0:n_box],scores[0][0:n_box], 0.3)
        print("list index: ", index)
        ratio = h_plate / image_resized.shape[0] 
        for idx in range(9):
            if idx not in index:
                continue
            y1, x1, y2, x2 = boxes[0][idx][:]    
            y1_ori, x1_ori, y2_ori, x2_ori = int(y1*ratio), int(x1*ratio), int(y2*ratio), int(x2*ratio)
            h, w = y2_ori - y1_ori, x2_ori - x1_ori
            if scores[0][idx] > threshold_score:

                if w_plate / h_plate < 2: # plate with 2 lines
                    # List down number and letter at thr bottom of plate
                    if h_plate * 0.8 > y1_ori > h_plate * 0.3: #and h > w and h_plate / 2.8 > h > h_plate / 5:
                        list_number.append([x1_ori, int(classes[0][idx] - 1)])
                        cv2.rectangle(image_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(image_resized, "{}".format(int(classes[0][idx] - 1)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),2)  
                elif h >= h_plate / 2 and h > w: # plate with 1 line only
                    list_number.append(int(classes[0][idx] - 1))

        list_number = sorted(list_number, key= lambda x: int(x[0]))
        numbers = [number for x, number in list_number]
        list_number_plates.append(numbers)

        #cv2.imshow("pl", image_resized)
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

def send_message_arduino(arduino_moduleSim, list_index_excel):
    now = datetime.now()
    thoigian = now.strftime("%H:%M:%S %d/%m/%Y")
    for idx in list_index_excel:
        if idx is None:
            continue
        message = "warning:" + list_number_phones[idx] + thoigian + list_num_plates_1[idx] + list_num_plates_excel[idx]
        print("Message: ", message)
        arduino_moduleSim.write(message.encode())
        print("-Sent data to Arduino-")
        time.sleep(0.1)
        for i in range(10):
            data = arduino_moduleSim.readline()
            data = data.decode("utf-8").rstrip('\r\n') 
            print(data)
        # while True:
        #     print(" data received: ",arduino_moduleSim.readline().decode("utf-8"))

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used for object detection.
PATH_TO_CKPT_vehicle = os.path.join(CWD_PATH,"Vehicle_ssd",'frozen_inference_graph.pb')
PATH_TO_CKPT_LicensePlate = os.path.join(CWD_PATH,"License_PLate",'frozen_inference_graph.pb')
PATH_TO_CKPT_NumberLetter = os.path.join(CWD_PATH,"NumberLetter",'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS_vehicle = os.path.join(CWD_PATH,'training','labelmap_vehicle.pbtxt')
PATH_TO_LABELS_LicensePlate = os.path.join(CWD_PATH,'training','labelmap_LicensePlate.pbtxt')
PATH_TO_LABELS_NumberLetter = os.path.join(CWD_PATH,'training','labelmap_NumberLetter.pbtxt')


# Load the label map.
label_map_vehicle = label_map_util.load_labelmap(PATH_TO_LABELS_vehicle)
categories_vehicle = label_map_util.convert_label_map_to_categories(label_map_vehicle, max_num_classes=2, use_display_name=True)
category_index_vehicle = label_map_util.create_category_index(categories_vehicle)

label_map_LicensePlate = label_map_util.load_labelmap(PATH_TO_LABELS_LicensePlate)
categories_LicensePlate = label_map_util.convert_label_map_to_categories(label_map_LicensePlate, max_num_classes=1, use_display_name=True)
category_index_LicensePlate = label_map_util.create_category_index(categories_LicensePlate)

label_map_NumberLetter = label_map_util.load_labelmap(PATH_TO_LABELS_NumberLetter)
categories_NumberLetter = label_map_util.convert_label_map_to_categories(label_map_NumberLetter, max_num_classes=1, use_display_name=True)
category_index_NumberLetter = label_map_util.create_category_index(categories_NumberLetter)

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

detection_graph_NumberLetter = tf.Graph()
with detection_graph_NumberLetter.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT_NumberLetter, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess_NumberLetter = tf.Session(graph=detection_graph_NumberLetter)
# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor_vehicle = detection_graph_vehicle.get_tensor_by_name('image_tensor:0')
image_tensor_LicensePlate = detection_graph_LicensePlate.get_tensor_by_name('image_tensor:0')
image_tensor_NumberLetter = detection_graph_NumberLetter.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes_vehicle = detection_graph_vehicle.get_tensor_by_name('detection_boxes:0')
detection_boxes_LicensePlate = detection_graph_LicensePlate.get_tensor_by_name('detection_boxes:0')
detection_boxes_NumberLetter = detection_graph_NumberLetter.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores_vehicle = detection_graph_vehicle.get_tensor_by_name('detection_scores:0')
detection_scores_LicensePlate = detection_graph_LicensePlate.get_tensor_by_name('detection_scores:0')
detection_scores_NumberLetter = detection_graph_NumberLetter.get_tensor_by_name('detection_scores:0')

detection_classes_vehicle = detection_graph_vehicle.get_tensor_by_name('detection_classes:0')
detection_classes_LicensePlate  = detection_graph_LicensePlate.get_tensor_by_name('detection_classes:0')
detection_classes_NumberLetter  = detection_graph_NumberLetter.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections_vehicle = detection_graph_vehicle.get_tensor_by_name('num_detections:0')
num_detections_LicensePlate = detection_graph_LicensePlate.get_tensor_by_name('num_detections:0')
num_detections_NumberLetter = detection_graph_NumberLetter.get_tensor_by_name('num_detections:0')

############## Set up connection between laptop and Arduino ############################
try:                                                                                   # 
    arduino_lighttraffic = serial.Serial("COM3", 9600 ,timeout=1)                      # 
    arduino_moduleSim = serial.Serial("COM6", 115200 ,timeout=1)                       #    
    print("Found out Arduino Uno device")                                              #
except:                                                                                #
    print("Please check the port")                                                     #
########################################################################################

########### LOAD EXCEL FILE #####################################################################
csv_sumary_path = "E:\\LicensePlateRecognition\\Autofill_Sumary\\Information_License_plate.xlsx"
data = pd.read_excel(csv_sumary_path, converters={'Biển số phần 2':str, 'Biển số phần 1':str, "Số điện thoại":str})
# Duyet 1 cot
list_num_plates_excel = list(data["Biển số phần 2"])
list_num_plates_1 = list(data["Biển số phần 1"])
list_number_phones = list(data["Số điện thoại"])

##################################################################################################
def check_number_plate(list_number_plates, list_num_plates_excel, list_num_plates_1, list_index_plate_excel_detected, list_full_number_plates):
    #list_full_number_plates = []
    for num_plate_predict in list_number_plates:
        flag_matching = False
        if num_plate_predict == False:
            #list_full_number_plates.append([])
            continue
        num_plate_predict = ''.join(map(str, num_plate_predict))
        if len(num_plate_predict) == 5:
            if num_plate_predict in list_num_plates_excel:
                    #print(list_num_plates_excel.index(num_plate_predict))
                    print("100%")
                    flag_matching = True 
                    index_excel = list_num_plates_excel.index(num_plate_predict)
            else:
                for num_plate in list_num_plates_excel:
                    for i in range(5):
                        if num_plate.replace(num_plate[i], '') == num_plate_predict.replace(num_plate_predict[i], ''):
                            print("80%")   
                            flag_matching = True      
                            index_excel = list_num_plates_excel.index(num_plate)   

        elif len(num_plate_predict) == 4: 
            for num_plate in list_num_plates_excel:
                for i in range(5):
                    if num_plate.replace(num_plate[i],'') == num_plate_predict:
                        print("80%")  
                        flag_matching = True 
                        index_excel = list_num_plates_excel.index(num_plate)
        elif len(num_plate_predict) == 3:
            for num_plate in list_num_plates_excel:
                for i in range(4):
                    if num_plate.replace(num_plate[i:i+2],'') == num_plate_predict:
                        print("60%")  
                        flag_matching = True 
                        index_excel = list_num_plates_excel.index(num_plate)

    
        if flag_matching and index_excel not in list_index_plate_excel_detected:
            list_index_plate_excel_detected.append(index_excel)
            list_full_number_plates.append(list_num_plates_1[index_excel] + list_num_plates_excel[index_excel])
        # else:
        #     list_full_number_plates.append([])
    return  list_index_plate_excel_detected, list_full_number_plates

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

    #####################################################################
    # list_index_plate_excel_detected: Contain index of the detected License Plate in excel file. 
    # Will reset when that vehicle pass red light or traffic light sign switch to green light.
    list_index_plate_excel_detected = []
    list_full_number_plates = []
    flag_red_light = True
    path = "E:\\project\\data_test4"
    files = [i for i in os.listdir(path) if i.endswith(".png") ]
    class_text = ["Background","motorbike", "car"]
    first_frame = True
    for filename in files:

        # time.sleep(0.5)
        # for i in range(5):
        #     data_light_traffic = arduino_lighttraffic.readline()
        #     data_light_traffic = data_light_traffic.decode("utf-8").rstrip('\r\n') 
        #     if data_light_traffic == "0": # red light
        #         print("Red mode")
        #         flag_red_light = True
        #     if data_light_traffic == "1":
        #         print("Green mode")
        #         flag_red_light = False

        start = time.time()
        print(filename)
        image_ori = cv2.imread(os.path.join(path, filename))
        if first_frame:
            slope, intercept, mask_line = detect_line(image_ori)
            first_frame = False
        
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

        list_License_Plate_box = detect_License_Plate(list_region_plate, 
                                                        vehicle_boxes,
                                                        sess_LicensePlate, 
                                                        detection_boxes_LicensePlate, 
                                                        detection_scores_LicensePlate, 
                                                        detection_classes_LicensePlate, 
                                                        num_detections_LicensePlate, 
                                                        image_tensor_LicensePlate,
                                                        threshold_score=0.003)   
        list_real_plate_mode, list_number_plates = recognize_plate(image_ori, 
                                                                list_License_Plate_box,
                                                                sess_NumberLetter,
                                                                detection_boxes_NumberLetter,
                                                                detection_scores_NumberLetter,
                                                                detection_classes_NumberLetter,
                                                                num_detections_NumberLetter,
                                                                image_tensor_NumberLetter,
                                                                threshold_score=0.2)

        list_index_plate_excel_detected, list_full_number_plates = check_number_plate(list_number_plates, list_num_plates_excel, list_num_plates_1, list_index_plate_excel_detected, list_full_number_plates)
        print(list_full_number_plates)
        list_traffic_violation_mode = check_pass_red_light(vehicle_boxes, slope, intercept)

        image_ori = visualize_image(image_ori, list_License_Plate_box, 
                                    vehicle_boxes, 
                                    list_full_number_plates, 
                                    flag_red_light,
                                    list_traffic_violation_mode,
                                    mask_line)
        print("index: ",list_index_plate_excel_detected)
        if flag_red_light and True in list_traffic_violation_mode:
            vuot_den_do_audio()
            #send_message_arduino(arduino_moduleSim, list_index_plate_excel_detected)
            list_full_number_plates = []
            list_index_plate_excel_detected = []
        if not flag_red_light:
            list_index_plate_excel_detected = []
            #list_full_number_plates = []
       
        cv2.imwrite("E:\\project\\result\\"+ filename, image_ori)
        cv2.imshow("", image_ori) 
        if cv2.waitKey(0) == 27:
            break
        print("timing: ",time.time() - start)
    cv2.destroyAllWindows()