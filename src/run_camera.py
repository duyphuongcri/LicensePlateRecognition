######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Duy-Phuong Dao
# E-mail: duypphuongcri@gmail.com
#############################################################################
"""
# count_vehicle_pass_on_line : so frame hinh xe can vach lien tiep
# slope, intercept           : hệ số góc và điểm cắt trục y của phương trình đường thẳng của vạch kẽ đường
# vehicle_boxes, classID = detect_vehicle() : vehicle_boxes-tọa độ y1,x1,y1,x2 của xe
                                              classID-nhãn label của xe
                                              detect_vehicle(): hàm detect xe
# list_traffic_violation_mode: [False, True] chứa thông tin có vượt đèn đỏ hay không, True: có vượt đèn đỏ
# list_index_plate_excel_detected : chứa vị trí của biển số xe detect đc trong file excel 
# list_full_number_plates : chứa thông tin biển số xe trích xuất được
"""

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
import traceback

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

import time
import argparse
from datetime import datetime
import serial

import base64
import json
import requests
################
from mailmerge import MailMerge
import xlrd
import pandas as pd
#####
from src.keras_utils 			import load_model
from glob 						import glob
from os.path 					import splitext, basename
from src.utils 					import im2single
from src.keras_utils 			import load_model, detect_lp
from src.label 					import Shape, writeShapes

def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))

def load_mailing_wordsheet_merge(excel_path, docx_path):
# Doc kiem tra duong dan
    if(os.path.isfile(docx_path)== False or os.path.isfile(excel_path)== False ):
        print("Cannot find input file.")
        sys.exit()

    # Doc file Word
    #print("Reading .docx file.")
    document = MailMerge(docx_path)
    #Get merge fields
    #print(document.get_merge_fields())
    #print("Done reading .docx file.")

    # Doc file excel
    #print("Reading .xlsx file.")
    book = xlrd.open_workbook(excel_path)
    print("Done read .xlsx file.")

    sheet_num = 0
    work_sheet = book.sheet_by_index(sheet_num)
    return work_sheet, document

def mailing_merge_save_violate(work_sheet, index_row ,document, output_path):
    
    if(os.path.isfile(output_path) == True):
        print("Output file already exists.")
        sys.exit()
    # Select the sheet that the data resids in
    finalList = []
    headers = []

    #get the total number of the rows
    # num_rows = work_sheet.nrows
    # Format required for mail merge is:
    # List [
    # {Dictrionaty},
    # {Dictrionaty},
    # ....
    # ]
    current_row = 0
    print("Preparing to merge.")
    # while current_row < num_rows:
    dictVal = dict()

    # Make header
    # if(current_row == 0):
    header_row = 0
    for col in range(work_sheet.ncols):
        headers.append(work_sheet.cell_value(header_row,col))

    # Update dictVal
    for col in range(work_sheet.ncols):
        dictVal.update({headers[col]:str(work_sheet.cell_value(index_row +1,col))})
    
    finalList.append(dictVal)
    #print(finalList)

    #print("Merge operation started.")
    document.merge_pages(finalList)
    #print("Saving output file.")

    time = datetime.now().strftime('%d_%m_%Y_%H_%M')
    bien_so = work_sheet.cell_value(index_row + 1,0) + work_sheet.cell_value(index_row + 1,1) 
    output_docx_name = bien_so + "_" + time
    # output_docx_name = '1'
    document.write(output_path + output_docx_name + ".docx")

    print("Operation complete successfully.")
##################
import pygame
pygame.mixer.init()
def phat_loa_vuot_den_do():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load('audio\\vuot_den_do.mp3')
        pygame.mixer.music.play()
def phat_loa_nhac_nho_dung_dung_vach():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load('audio\\dung_phuong_tien_dung_vach_vi_dinh.mp3')
        pygame.mixer.music.play()
def phat_loa_khong_doi_mu_bao_hiem():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load('audio\\no_helmet.mp3')
        pygame.mixer.music.play()
def phat_loa_khong_doi_mbh_vuot_den_do():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load('audio\\vuot_den_do_no_helmet.mp3')
        pygame.mixer.music.play()
##################
def post_API_and_export_report_and_send_message_arduino(image_ori, arduino_moduleSim, list_index, work_sheet, document, output_path, lastest_number_plate, mode):
    now = datetime.now()
    thoigian = now.strftime("%H:%M:%S %d/%m/%Y")
    image_id = now.strftime("%H_%M_%S") + ".png"
    cv2.imwrite("Hinh_anh_vi_pham\\{}".format(image_id), image_ori)
    for idx in list_index:
        if idx is None:
            continue
        bien_so = list_num_plates_1[idx] + list_num_plates_excel[idx]
        if bien_so == lastest_number_plate:
            continue
        lastest_number_plate = bien_so
        # gui tin nhan
        # if mode == 0: # vuot den do  + ko doi mbh
        #     message = "hailoi: " + list_number_phones[idx] + thoigian + bien_so
        #     print("Message: ", message)
        #     arduino_moduleSim.write(message.encode())
        #     print("-Sent data to Arduino-")
        # elif mode == 1: # vuot den do
        #     message = "redlight" + list_number_phones[idx] + thoigian + bien_so
        #     print("Message: ", message)
        #     arduino_moduleSim.write(message.encode())
        #     print("-Sent data to Arduino-")
        # elif mode == 2: # ko doi mu bao hiem
        #     message = "nohelmet" + list_number_phones[idx] + thoigian + bien_so
        #     print("Message: ", message)
        #     arduino_moduleSim.write(message.encode())
        #     print("-Sent data to Arduino-")
        # time.sleep(0.05)
        # for i in range(10):
        #     data = arduino_moduleSim.readline()
        #     data = data.decode("utf-8").rstrip('\r\n') 
        #     print(data)

        # xuat bien ban xu phat
        mailing_merge_save_violate(work_sheet, idx ,document, output_path)

        print("Send data to WebApp")
        # defining the api-endpoint  
        #API_URL = "http://192.168.137.167:8000/newfails"
        API_URL = "http://vuotdendotpbentre.ddns.net:8000/newfails"
        # your API key here 
        #API_KEY = "XXXXXXXXXXXXXXXXX"
        
        with open('Hinh_anh_vi_pham\\{}'.format(image_id), 'rb') as f:
            img_str = 'data:image/png;base64,'
            img_str += base64.b64encode(f.read()).decode('utf-8')

            # data to be sent to api 
            data = json.dumps({'Blate': bien_so, 
                            'date':now.strftime("%Y/%m/%d"), 
                            'time':now.strftime("%H:%M:%S"), 
                            'img':img_str,
                            'type': str(mode)})
            headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
            
            # sending post request and saving response as response object 
            r = requests.post(url=API_URL, data=data, headers=headers) 
            
            # extracting response text  
            pastebin_url = r.text 
            print("The pastebin URL is:%s"%pastebin_url) 
    return lastest_number_plate

#################

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

def check_pass_line(vehicle_boxes, slope, intercept, count_vehicle_override_line, count_vehicle_front_line):
    list_vehicle_pass_line_mode =[] # 0: binh thuong ; 1: can vach; 2: vuot den do
    for i, box in enumerate(vehicle_boxes):
        if len(box) == 0:
            list_vehicle_pass_line_mode.append(0)
            continue

        h, w = box[2] - box[0], box[3] - box[1]
        coor_x = box[1] + w*0.5
        coor_y = box[0] + h*0.5
        if slope*coor_x + intercept - coor_y > 0 and slope*coor_x + intercept - box[2] > 0: # qua vach
            count_vehicle_front_line = 0
            count_vehicle_override_line = 0
            list_vehicle_pass_line_mode.append(2)
        elif slope*coor_x + intercept - coor_y > 0 and slope*coor_x + intercept - box[2] <= 0: # can vach
            count_vehicle_front_line = 0
            count_vehicle_override_line =  count_vehicle_override_line + 1 
            #print(count_vehicle_override_line)
            if count_vehicle_override_line > 5: #
                list_vehicle_pass_line_mode.append(1)
            else:
                list_vehicle_pass_line_mode.append(0)
        else: # dung dung vach
            count_vehicle_front_line += 1
            list_vehicle_pass_line_mode.append(0)
            count_vehicle_override_line = 0

    return list_vehicle_pass_line_mode, count_vehicle_override_line, count_vehicle_front_line

def detect_line(img):
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bw = cv2.threshold(img_gray, 170, 255, cv2.THRESH_BINARY)
    #cv2.imshow("sdasd", img_bw)
    lines = cv2.HoughLinesP(img_bw, rho=1, theta=np.pi/180, threshold=300, minLineLength=100, maxLineGap=30)
    mask = np.zeros(img_bw.shape, dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2:
                continue
            slope = (y2 - y1)/(x2 - x1)
            intercept = y1 - slope*x1
            if -0.1 < slope < 0.1 and 350 < intercept < 400:
                #print(" 0 =  {0:.2f}*x + {1:.2f} - y".format(slope, intercept))
                
                cv2.line(mask, (0, int(slope*0 + intercept)), (img_gray.shape[1], int(slope*img_gray.shape[1] + intercept)), (255), 10)     
                # draw mask of line
                ret, img_bw_150 = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)
                mask = cv2.bitwise_and(mask, img_bw_150)
                img[:, :, 0] = np.where(mask == 255 ,100, img[:, :, 0])
                img[:, :, 1] = np.where(mask == 255 ,91, img[:, :, 1])
                img[:, :, 2] = np.where(mask == 255 ,210, img[:, :, 2]) 
                return slope, intercept, mask, img

    return 0, 0, mask, img


def visualize_image(image_ori, list_license_plate_box, list_helmet_box, vehicle_boxes, list_full_number_plates, flag_red_light, list_wear_helmet, list_vehicle_pass_line_mode, mask_line, list_real_plate_mode):

    # draw mask of line
    ret, img_bw_150 = cv2.threshold(cv2.cvtColor(image_ori, cv2.COLOR_BGR2GRAY), 120, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask_line, img_bw_150)
    image_ori[:, :, 0] = np.where(mask == 255 ,100, image_ori[:, :, 0])
    image_ori[:, :, 1] = np.where(mask == 255 ,91, image_ori[:, :, 1])
    image_ori[:, :, 2] = np.where(mask == 255 ,210, image_ori[:, :, 2])
 
    # # draw vehicles
    for i,box in enumerate(vehicle_boxes):
        cv2.circle(image_ori, (int((box[1]+box[3])/2), int((box[0]+box[2])/2)), 5,(255,255,255), -1)  
        cv2.rectangle(image_ori, (box[1], box[2]),(box[1] + 170, box[2] + 25), (0,255,0), -1)  
        cv2.rectangle(image_ori, (box[1], box[0]),(box[3], box[2]), (0,255,0), 2)
        cv2.putText(image_ori, str(class_text[classID[i]]), (box[1], box[2] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),2)
        if list_vehicle_pass_line_mode[i] != 0 and flag_red_light:
            image_ori[box[0]: box[2], box[1]: box[3], 2] = 250
            cv2.rectangle(image_ori, (box[1], box[0]),(box[3], box[2]), (0,0,255), 2)
        else:
            cv2.rectangle(image_ori, (box[1], box[0]),(box[3], box[2]), (0,255,0), 2)

    # # # # # Draw license plate
    for i, box in enumerate(list_license_plate_box):
        if box is None:
            continue
        if list_real_plate_mode[i]:
            cv2.rectangle(image_ori, (box[1], box[0] - 30),(box[1] + 65, box[0]), (0,255,0), -1) 
            cv2.putText(image_ori, "Plate", (box[1], box[0] - 5 ), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),2) 
            cv2.rectangle(image_ori, (box[1], box[0]),(box[3], box[2]), (0,255,0), 2)   
        if len(list_full_number_plates) > 0: # display license plate number
            cv2.rectangle(image_ori, (box[1], box[2]),(box[1] + 180, box[2] + 40), (0,255,0), -1)  
            cv2.putText(image_ori, list_full_number_plates[0] , (box[1], box[2] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),2) #(240, 50, 50)
    ## Display helmet #######
    for i, box in enumerate(list_helmet_box):
        if box is None:
            continue 
        cv2.rectangle(image_ori, (box[1], box[2]),(box[1] + 90, box[2] + 25), (0,255,0), -1)  
        cv2.putText(image_ori, "Helmet", (box[1], box[2] + 20 ), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),2)  
        cv2.rectangle(image_ori, (box[1], box[0]),(box[3], box[2]), (0,255,0), 2)             

    # # Visualize date time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    cv2.putText(image_ori, dt_string, (10, 1025), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0),2)

    # display location
    cv2.putText(image_ori, """Location: 10*14'09.9"N 106*22'20.6"E""", (10, 1065), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0),2) 

    # display traffic light mode
    cv2.putText(image_ori, "Traffic Light Status: ", (10, 980), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0, 255, 0),2) 
    if flag_red_light:
        cv2.circle(image_ori, (480,970), 30, (0,0,255), -1)# den do
        # pass red light or not
        if 2 in list_vehicle_pass_line_mode and (2 in list_wear_helmet and 1 not in list_wear_helmet): # vuot den do va ko doi mbn
            cv2.putText(image_ori, "Traffic Violation & No Helmet", (800, 1060), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0, 0, 255),3) 
        elif 2 in list_vehicle_pass_line_mode:
            cv2.putText(image_ori, "Traffic Violation", (1010, 1060), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255),3) 
    else:
        cv2.circle(image_ori, (480,970), 30, (0,255,0), -1) # xanh
        if (2 in list_wear_helmet and 1 not in list_wear_helmet): #ko doi mbh
            cv2.putText(image_ori, "No Helmet", (1010, 1060), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0, 0, 255),2) 
    
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
                    list_plate_region,
                    classID,
                    sess, 
                    detection_boxes, 
                    detection_scores,
                    detection_classes, 
                    num_detections, 
                    image_tensor, 
                    threshold_score):
    #######################################
    # list_real_plate_mode : False mean this box is not a plate, 
    #
    #
    list_real_plate_mode = []
    list_number_plates = []
    for i, plate_region in enumerate(list_plate_region):
        if plate_region is None:
            list_real_plate_mode.append(False)
            list_number_plates.append(None)
            continue
        list_number = []
        h_plate, w_plate = plate_region.shape[:2]
        #print("Shape of plate box: ",h_plate, w_plate)
        # img_gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        # mean, stv = cv2.meanStdDev(img_gray)
        # if mean > 90 and w_plate < 100 and h_plate < 100 and h_plate < w_plate:
        #     list_real_plate_mode.append(True)
        # else:
        #     list_real_plate_mode.append(False)
        #     continue
        list_real_plate_mode.append(True)
        
        # pd = w_plate - h_plate
        # if pd > 0:
        #     padding = [(0, pd), (0, 0),(0,0)]
        # else:
        #     padding = [(0, 0), (0, -pd),(0,0)]
        # image_padding = np.pad(plate_region, padding, mode='constant', constant_values=0)
        # image_resized = cv2.resize(image_padding,(300, 300), interpolation=cv2.INTER_AREA)
        image_expanded = np.expand_dims(plate_region, axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        n_box = 12
        boxes[0][:,0] = boxes[0][:,0]*h_plate
        boxes[0][:,1] = boxes[0][:,1]*w_plate
        boxes[0][:,2] = boxes[0][:,2]*h_plate
        boxes[0][:,3] = boxes[0][:,3]*w_plate
        # perform non-maximum suppression on the bounding boxes
        index = non_max_suppression(boxes[0][0:n_box],scores[0][0:n_box], 0.3)
        #print("list index: ", index)
        ratio = 1
        for idx in range(n_box):
            if idx not in index:
                continue
            y1, x1, y2, x2 = boxes[0][idx][:]    
            y1_ori, x1_ori, y2_ori, x2_ori = int(y1*ratio), int(x1*ratio), int(y2*ratio), int(x2*ratio)
            h, w = y2_ori - y1_ori, x2_ori - x1_ori
            if scores[0][idx] > threshold_score:
                if np.any(plate_region[110:, :] != 0): # plate with 2 lines
                    # List down number and letter at thr bottom of plate
                    if h_plate * 0.8 > y1_ori > h_plate * 0.3: #and h > w and h_plate / 2.8 > h > h_plate / 5:
                        list_number.append([x1_ori, int(classes[0][idx] - 1)])
                        cv2.rectangle(plate_region, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(plate_region, "{}".format(int(classes[0][idx] - 1)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),2)  
                    else:
                        cv2.rectangle(plate_region, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(plate_region, "{}".format(int(classes[0][idx] - 1)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),2)
                else: # plate with 1 line only
                    list_number.append([x1_ori, int(classes[0][idx] - 1)])
                    cv2.rectangle(plate_region, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(plate_region, "{}".format(int(classes[0][idx] - 1)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),2)  

        list_number = sorted(list_number, key= lambda x: int(x[0]))
        numbers = [number for x, number in list_number]
        list_number_plates.append(numbers)
        cv2.imshow("pl", plate_region)
    return list_real_plate_mode, list_number_plates
def crop_top_bottom_region_vehicle(input_image, vehicle_boxes, classID, offset):
    list_bottom_region_vehicle = []
    list_top_region_vehicle = []
    for i, box in enumerate(vehicle_boxes): 
        h, w = box[2] - box[0], box[3] -  box[1]
        # Crop bot region ( it contains plate region)
        if box[0] < input_image.shape[0]/2 < box[2] and w/h < 2 and classID[i] == 2:
            bottom_region = input_image[int(box[0] + h/2): int(box[2]), int(box[1]): int(box[3])]
        elif classID[i] == 1:
            bottom_region = input_image[int(box[0]): int(box[2]), int(box[1]): int(box[3])]
        else:
            bottom_region = None
        # Crop top region (it contains helmet)
        if box[2] > input_image.shape[0]*0.6 and w/h < 2 and classID[i] == 2: # this case for motorbike
            top_region = input_image[int(abs(box[0] - offset)): int(box[0] + w - offset), int(box[1]): int(box[3])]
            #cv2.imshow("top", top_region)
        else:
            top_region = None
        list_bottom_region_vehicle.append(bottom_region)
        list_top_region_vehicle.append(top_region)
    return list_bottom_region_vehicle, list_top_region_vehicle
        
        
def crop_vehicle_region(input_image, vehicle_boxes, classID):
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
        # if img_region_plate.shape[0] > 300:
        #     img_region_plate = img_region_plate[0:300, 0:300]

        list_region_plate.append(img_region_plate)   
    return list_region_plate

def license_plate_detector(list_bottom_region_vehicle, vehicle_boxes, classID, wpod_net, lp_threshold=0.5):
    list_license_plate_box = []
    list_plate_region = []
    for i, bottom_region_vehicle in enumerate(list_bottom_region_vehicle):
        if bottom_region_vehicle is None:
            list_license_plate_box.append(None)
            list_plate_region.append(None)
            continue
        ratio = float(max(bottom_region_vehicle.shape[:2]))/min(bottom_region_vehicle.shape[:2])
        side  = int(ratio*288.) #288
        bound_dim = min(side + (side%(2**4)),608) #608
        #print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))
        ratio_w_h = bottom_region_vehicle.shape[1]/bottom_region_vehicle.shape[0]
        Llp, LlpImgs = detect_lp(wpod_net,im2single(bottom_region_vehicle),bound_dim,2**4,(300,220),lp_threshold, ratio_w_h)

        if len(LlpImgs):
            Ilp = LlpImgs[0]
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
            Ilp = (Ilp*255).astype(np.uint8)
            x1, y1 = int(np.amin(Llp[0].pts, 1)[0]*bottom_region_vehicle.shape[1]), int(np.amin(Llp[0].pts, 1)[1]*bottom_region_vehicle.shape[0])
            x2, y2 = int(np.amax(Llp[0].pts, 1)[0]*bottom_region_vehicle.shape[1]), int(np.amax(Llp[0].pts, 1)[1]*bottom_region_vehicle.shape[0])
            x1_ori, y1_ori = x1 + vehicle_boxes[i][1], y1 + bottom_region_vehicle.shape[0] + vehicle_boxes[i][0]
            x2_ori, y2_ori = x2 + vehicle_boxes[i][1], y2 + bottom_region_vehicle.shape[0] + vehicle_boxes[i][0]
            if Ilp.shape[0] == 110:
                x1_ori = int(x1_ori - (x2_ori - x1_ori)/0.6*0.4)
                padding = [(0, 110), (0, 0),(0,0)]
                Ilp = np.pad(Ilp, padding, mode='constant', constant_values=0)
            if classID[i] == 1: #Car
                y1_ori, y2_ori = y1_ori - bottom_region_vehicle.shape[0], y2_ori - bottom_region_vehicle.shape[0]
            list_license_plate_box.append([y1_ori, x1_ori, y2_ori, x2_ori])
            list_plate_region.append(Ilp)
            #cv2.imshow("plate region: ", Ilp) ##### IMSHOW ###########
        else:
            list_license_plate_box.append(None)
            list_plate_region.append(None)
    return list_license_plate_box, list_plate_region

def helmet_detector(list_top_region_vehicle, 
                    vehicle_boxes,   
                    list_wear_helmet,     
                    sess,                   
                    detection_boxes, 
                    detection_scores,
                    detection_classes, 
                    num_detections, 
                    image_tensor, 
                    offset,
                    threshold_score):
    # list_wear_helmet   0: normal, 1: wear helmet, 2: dont wear helmet
    list_helmet_box = []
    for i, top_region_vehicle in enumerate(list_top_region_vehicle):
        if top_region_vehicle is None:
            list_helmet_box.append(None)
            list_wear_helmet.append(0)
            continue
        img_resized = cv2.resize(top_region_vehicle, (300,300))
        image_expanded = np.expand_dims(img_resized, axis=0)
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        for j in range(1): # ko dc doi
            if scores[0][j] > threshold_score:
                y1, x1, y2, x2 = int(boxes[0][j][0]*img_resized.shape[0]), int(boxes[0][j][1]*img_resized.shape[1]), int(boxes[0][j][2]*img_resized.shape[0]), int(boxes[0][j][3]*img_resized.shape[1])
                ratio = top_region_vehicle.shape[0]/img_resized.shape[0] 
                y1_ori, x1_ori, y2_ori, x2_ori = int(y1*ratio + vehicle_boxes[i][0] - offset), int(x1*ratio + vehicle_boxes[i][1]), int(y2*ratio + vehicle_boxes[i][0] - offset), int(x2*ratio + vehicle_boxes[i][1])
                list_wear_helmet.append(1)
                list_helmet_box.append([y1_ori, x1_ori, y2_ori, x2_ori])
                #cv2.imshow("helmet", top_region_vehicle[y1:y2, x1:x2])
            else:
                list_wear_helmet.append(2)
                list_helmet_box.append(None)                
    return list_helmet_box, list_wear_helmet
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
                y1_Plate_ori, x1_Plate_ori, y2_Plate_ori, x2_Plate_ori = int(y1*ratio + vehicle_boxes[i][0]), int(x1*ratio + vehicle_boxes[i][1]), int(y2*ratio + + vehicle_boxes[i][0]), int(x2*ratio + vehicle_boxes[i][1])
                list_License_Plate_box.append([y1_Plate_ori, x1_Plate_ori, y2_Plate_ori, x2_Plate_ori])
            else:
                list_License_Plate_box.append([])
 
    return list_License_Plate_box
            
def detect_vehicle(input_image, sess, detection_boxes, detection_scores, detection_classes, num_detections, image_tensor, threshold_score):
    h_ori, w_ori = input_image.shape[:2]

    ##Resize input image
    image = cv2.resize(input_image, (640,480))
    image_expanded = np.expand_dims(image, axis=0)
     
    #image = input_image
    #image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    vehicle_boxes = []
    classID = []
    for i in range(1):
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

def send_message_arduino(arduino_moduleSim, list_index_excel, lastest_number_plate, mode):
    # mode = 0: ko doi mu bao hiem; mode = 1: vuot den do
    now = datetime.now()
    thoigian = now.strftime("%H:%M:%S %d/%m/%Y")
    for idx in list_index_excel:
        if idx is None:
            continue
        bien_so = list_num_plates_1[idx] + list_num_plates_excel[idx]

        if mode == 0:
            message = "nohelmet" + list_number_phones[idx] + thoigian + bien_so
            print("Message: ", message)
            arduino_moduleSim.write(message.encode())
            print("-Sent data to Arduino-")
        elif mode == 1:
            message = "redlight" + list_number_phones[idx] + thoigian + bien_so
            print("Message: ", message)
            arduino_moduleSim.write(message.encode())
            print("-Sent data to Arduino-")
        # time.sleep(0.05)
        # for i in range(10):
        #     data = arduino_moduleSim.readline()
        #     data = data.decode("utf-8").rstrip('\r\n') 
        #     print(data)


# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used for object detection.
PATH_TO_CKPT_vehicle = os.path.join(CWD_PATH,"models\\Vehicle_ssd",'frozen_inference_graph.pb')
PATH_TO_CKPT_helmet = os.path.join(CWD_PATH,"models\\Helmet",'frozen_inference_graph.pb')
PATH_TO_CKPT_NumberLetter = os.path.join(CWD_PATH,"models\\NumberLetter",'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS_vehicle = os.path.join(CWD_PATH,'labels','labelmap_vehicle.pbtxt')
PATH_TO_LABELS_helmet = os.path.join(CWD_PATH,'labels','labelmap_helmet.pbtxt')
PATH_TO_LABELS_NumberLetter = os.path.join(CWD_PATH,'labels','labelmap_NumberLetter.pbtxt')


# Load the label map.
label_map_vehicle = label_map_util.load_labelmap(PATH_TO_LABELS_vehicle)
categories_vehicle = label_map_util.convert_label_map_to_categories(label_map_vehicle, max_num_classes=2, use_display_name=True)
category_index_vehicle = label_map_util.create_category_index(categories_vehicle)

label_map_helmet = label_map_util.load_labelmap(PATH_TO_LABELS_helmet)
categories_helmet = label_map_util.convert_label_map_to_categories(label_map_helmet, max_num_classes=1, use_display_name=True)
category_index_helmet = label_map_util.create_category_index(categories_helmet)

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

detection_graph_helmet = tf.Graph()
with detection_graph_helmet.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT_helmet, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess_helmet = tf.Session(graph=detection_graph_helmet)

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
image_tensor_helmet = detection_graph_helmet.get_tensor_by_name('image_tensor:0')
image_tensor_NumberLetter = detection_graph_NumberLetter.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes_vehicle = detection_graph_vehicle.get_tensor_by_name('detection_boxes:0')
detection_boxes_helmet = detection_graph_helmet.get_tensor_by_name('detection_boxes:0')
detection_boxes_NumberLetter = detection_graph_NumberLetter.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores_vehicle = detection_graph_vehicle.get_tensor_by_name('detection_scores:0')
detection_scores_helmet = detection_graph_helmet.get_tensor_by_name('detection_scores:0')
detection_scores_NumberLetter = detection_graph_NumberLetter.get_tensor_by_name('detection_scores:0')

detection_classes_vehicle = detection_graph_vehicle.get_tensor_by_name('detection_classes:0')
detection_classes_helmet  = detection_graph_helmet.get_tensor_by_name('detection_classes:0')
detection_classes_NumberLetter  = detection_graph_NumberLetter.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections_vehicle = detection_graph_vehicle.get_tensor_by_name('num_detections:0')
num_detections_helmet = detection_graph_helmet.get_tensor_by_name('num_detections:0')
num_detections_NumberLetter = detection_graph_NumberLetter.get_tensor_by_name('num_detections:0')



########### LOAD EXCEL FILE #####################################################################
csv_sumary_path = os.path.join(CWD_PATH,"Information_License_plate.xlsx")
data = pd.read_excel(csv_sumary_path, converters={'Bien_so_phan_2':str, 'Bien_so_phan_1':str, "So_DT":str})
# Duyet 1 cot
list_num_plates_excel = list(data["Bien_so_phan_2"])
list_num_plates_1 = list(data["Bien_so_phan_1"])
list_number_phones = list(data["So_DT"])

##################################################################################################
def check_number_plate(list_number_plates, list_num_plates_excel, list_num_plates_1, list_index_plate_excel_detected, list_full_number_plates):
    for num_plate_predict in list_number_plates:
        flag_matching = False
        if num_plate_predict is None:
            continue
        num_plate_predict = ''.join(map(str, num_plate_predict))
        if len(num_plate_predict) == 5:
            if num_plate_predict in list_num_plates_excel:
                # print(list_num_plates_excel.index(num_plate_predict))
                # print("100%")
                flag_matching = True 
                index_excel = list_num_plates_excel.index(num_plate_predict)
            else:                
                for num_plate in list_num_plates_excel:
                    if np.isnan(float(num_plate)):
                        continue
                    for i in range(5):
                        if num_plate[:i] + num_plate[i+1:5] == num_plate_predict[:i] + num_plate_predict[i+1:5]:
                            #print("80%")   
                            flag_matching = True      
                            index_excel = list_num_plates_excel.index(num_plate)   
                            break
                    if flag_matching:
                        break
        elif len(num_plate_predict) == 4: 
            for num_plate in list_num_plates_excel:
                if np.isnan(float(num_plate)):
                    continue
                if num_plate_predict in list_num_plates_excel:
                    flag_matching = True 
                    index_excel = list_num_plates_excel.index(num_plate_predict)
                    break
                for i in range(5):
                    if num_plate[:int(i)] + num_plate[int(i+1):5] == num_plate_predict:
                        # print("80%")  
                        flag_matching = True 
                        index_excel = list_num_plates_excel.index(num_plate)
                        break
                if flag_matching:
                    break
        # elif len(num_plate_predict) == 3:
        #     for num_plate in list_num_plates_excel:
        #         for i in range(4):
        #             if num_plate[:i] + num_plate[i+2:5] == num_plate_predict:
        #                 # print("60%")  
        #                 flag_matching = True 
        #                 index_excel = list_num_plates_excel.index(num_plate)
        #                 break
        #         if flag_matching:
        #             break

        if flag_matching and index_excel not in list_index_plate_excel_detected:
            list_index_plate_excel_detected.append(index_excel)
            list_full_number_plates.append(list_num_plates_1[index_excel] + list_num_plates_excel[index_excel])

    return  list_index_plate_excel_detected, list_full_number_plates
    
if __name__=="__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Detect vehicle and License Plate using Object Detection .')
    # # parser.add_argument("command",
    # #                     metavar="<command>",
    # #                     help="'train' or 'test'")
    parser.add_argument('--port_sim', required=False,
                        metavar="",
                        help='Port number of Arduino( connect with module sim)')
    parser.add_argument('--port_light', required=False,
                        metavar="",
                        help='Port number of Arduino( connect with traffic light sign)')

    args = parser.parse_args()

    ############## Set up connection between laptop and Arduino ############################
    try:                                                                                   #
        arduino_moduleSim = serial.Serial(args.port_sim, 115200 ,timeout=1)                #    
        arduino_lighttraffic = serial.Serial(args.port_light, 9600 ,timeout=1)             # 
        print("Found out Arduino Uno devices")                                             #
    except:                                                                                #
        print("Please check the port again")                                               #
    ########################################################################################

    # Initialize Camera
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) #3 1280
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) #4 720
    list_index_plate_excel_detected = []
    list_full_number_plates = []
    list_wear_helmet = []
    flag_red_light = False
    class_text = ["Background","car", "motorbike"]
    first_frame = True
    n = 0 # save frame
    flag_predict_virtual_images = True # predict virtual image first
    count_vehicle_override_line = 0
    count_vehicle_front_line = 0
    lastest_number_plate = []

    docx_path = os.path.join(CWD_PATH,"Bien_ban_vi_pham.docx")
    excel_path = os.path.join(CWD_PATH,"Information_License_plate.xlsx")
    output_path = "Bien_ban\\"

    work_sheet, document = load_mailing_wordsheet_merge(excel_path, docx_path)
    # load model plate detection
    wpod_net_path = "wpod-net_update1.h5"
    wpod_net = load_model(wpod_net_path)
    slope, intercept = 0.01, 400
    mask_line = np.zeros((1080,1440), dtype=np.uint8)
    path = "E:\\project\\test3"
    files = [i for i in os.listdir(path) if i.endswith(".png")]
    for filename in files:
    # while True:
        if flag_predict_virtual_images:
            print("loadding model...")
            img = cv2.imread("test.png")
            #img = cv2.imread(os.path.join(path, filename))
            vehicle_boxes, classID = detect_vehicle(img, 
                                                    sess_vehicle, 
                                                    detection_boxes_vehicle, 
                                                    detection_scores_vehicle, 
                                                    detection_classes_vehicle, 
                                                    num_detections_vehicle, 
                                                    image_tensor_vehicle,
                                                    threshold_score=0.8)
            # ## Crop region of interest of License Plate to pass another model which is used to detect License Plate 
            list_bottom_region_vehicle, list_top_region_vehicle = crop_top_bottom_region_vehicle(img, vehicle_boxes, classID, offset=20)     
            list_license_plate_box, list_plate_region = license_plate_detector(list_bottom_region_vehicle,
                                                                                vehicle_boxes,
                                                                                classID,
                                                                                wpod_net,
                                                                                lp_threshold=0.8)
            _, _ = helmet_detector(list_top_region_vehicle, 
                                    vehicle_boxes,
                                    list_wear_helmet,
                                    sess_helmet, 
                                    detection_boxes_helmet, 
                                    detection_scores_helmet, 
                                    detection_classes_helmet, 
                                    num_detections_helmet, 
                                    image_tensor_helmet,
                                    offset=20,
                                    threshold_score=0.95)
                                                   
            # Trich xuat cac ki tu trong bien so xe
            _, _ = recognize_plate(img, 
                                    list_plate_region,
                                    classID,
                                    sess_NumberLetter,
                                    detection_boxes_NumberLetter,
                                    detection_scores_NumberLetter,
                                    detection_classes_NumberLetter,
                                    num_detections_NumberLetter,
                                    image_tensor_NumberLetter,
                                    threshold_score=0.05)

            flag_predict_virtual_images = False  
            list_wear_helmet = []
            print("Done loading model")   
        
        # read data from Arduino, "0" :  red light mode
        # data_light_traffic = arduino_lighttraffic.readline()
        # data_light_traffic = data_light_traffic.decode("utf-8").rstrip('\r\n') 
        # if data_light_traffic == "0": # red light
        #     flag_red_light = True
        # if data_light_traffic == "1":
            # flag_red_light = False

        start = time.time()
        #ret, image_ori = cam.read()
        # cv2.imwrite("E:\\project\\test7\\image%06d.png"%n, image_ori)
        # n += 1
        image_ori = cv2.imread(os.path.join(path, filename))
        #image_ori = cv2.imread("E:\\project\\test\\image0222.png")
        image_ori = image_ori[:, 240:1680]
        if image_ori is None:
            continue

        if first_frame:
            slope, intercept, mask_line, image_ori = detect_line(image_ori)  
            #slope, intercept = 0.01, 400
        else:   

            vehicle_boxes, classID = detect_vehicle(image_ori, 
                                                    sess_vehicle, 
                                                    detection_boxes_vehicle, 
                                                    detection_scores_vehicle, 
                                                    detection_classes_vehicle, 
                                                    num_detections_vehicle, 
                                                    image_tensor_vehicle,
                                                    threshold_score=0.9)
            # ## Crop region of interest of License Plate to pass another model which is used to detect License Plate 
            list_bottom_region_vehicle, list_top_region_vehicle = crop_top_bottom_region_vehicle(image_ori, vehicle_boxes, classID, offset=20)
            # Phát hiện vị trí biển số xe trong ảnh
            list_license_plate_box, list_plate_region = license_plate_detector(list_bottom_region_vehicle,
                                                                                vehicle_boxes,
                                                                                classID,
                                                                                wpod_net,
                                                                                lp_threshold=0.8) 
            list_helmet_box, list_wear_helmet = helmet_detector(list_top_region_vehicle, 
                                                                vehicle_boxes,
                                                                list_wear_helmet,
                                                                sess_helmet, 
                                                                detection_boxes_helmet, 
                                                                detection_scores_helmet, 
                                                                detection_classes_helmet, 
                                                                num_detections_helmet, 
                                                                image_tensor_helmet,
                                                                offset=20,
                                                                threshold_score=0.95)
            # Trich xuat cac ki tu trong bien so xe
            if len(list_full_number_plates) == 0:
                list_real_plate_mode, list_number_plates = recognize_plate(image_ori, 
                                                                        list_plate_region,
                                                                        classID,
                                                                        sess_NumberLetter,
                                                                        detection_boxes_NumberLetter,
                                                                        detection_scores_NumberLetter,
                                                                        detection_classes_NumberLetter,
                                                                        num_detections_NumberLetter,
                                                                        image_tensor_NumberLetter,
                                                                        threshold_score=0.05)

                list_index_plate_excel_detected, list_full_number_plates = check_number_plate(list_number_plates, list_num_plates_excel, list_num_plates_1, list_index_plate_excel_detected, list_full_number_plates)
            # Kiem tra xem xe co vuot den do hay chua?
            list_vehicle_pass_line_mode, count_vehicle_override_line, count_vehicle_front_line = check_pass_line(vehicle_boxes, slope, intercept, count_vehicle_override_line, count_vehicle_front_line)

            # Hien thi ket qua
            image_ori = visualize_image(image_ori, 
                                        list_license_plate_box, 
                                        list_helmet_box,
                                        vehicle_boxes, 
                                        list_full_number_plates, 
                                        flag_red_light,
                                        list_wear_helmet,
                                        list_vehicle_pass_line_mode,
                                        mask_line,
                                        list_real_plate_mode)

            if flag_red_light and 2 in list_vehicle_pass_line_mode and (2 in list_wear_helmet and 1 not in list_wear_helmet) : # vuot den do + ko doi mu bao hiem
                lastest_number_plate = post_API_and_export_report_and_send_message_arduino(image_ori, 
                                                                                            arduino_moduleSim,
                                                                                            list_index_plate_excel_detected, 
                                                                                            work_sheet, 
                                                                                            document, 
                                                                                            output_path, 
                                                                                            lastest_number_plate, 
                                                                                            mode=0) 
                phat_loa_khong_doi_mbh_vuot_den_do() # phat loa
                list_full_number_plates, list_index_plate_excel_detected, list_wear_helmet = [], [], []  
                print("vuot den do + ko doi mu") 
            elif flag_red_light and 2 in list_vehicle_pass_line_mode and 1 in list_wear_helmet: # vuot den do + doi mu bh
                lastest_number_plate = post_API_and_export_report_and_send_message_arduino(image_ori, 
                                                                                            arduino_moduleSim,
                                                                                            list_index_plate_excel_detected, 
                                                                                            work_sheet, 
                                                                                            document, 
                                                                                            output_path, 
                                                                                            lastest_number_plate, 
                                                                                            mode=1) 
                phat_loa_vuot_den_do() # phat loa
                list_full_number_plates, list_index_plate_excel_detected, list_wear_helmet = [], [], []   
                print("vuot den do")
            elif ((flag_red_light and 2 not in list_vehicle_pass_line_mode and count_vehicle_front_line > 6) or not flag_red_light) and 2 in list_wear_helmet and 1 not in list_wear_helmet: # ko doi mu bh
                lastest_number_plate = post_API_and_export_report_and_send_message_arduino(image_ori, 
                                                                                            arduino_moduleSim,
                                                                                            list_index_plate_excel_detected, 
                                                                                            work_sheet, 
                                                                                            document, 
                                                                                            output_path, 
                                                                                            lastest_number_plate, 
                                                                                            mode=2)
                phat_loa_khong_doi_mu_bao_hiem() 
                list_full_number_plates, list_index_plate_excel_detected, list_wear_helmet, count_vehicle_front_line = [], [], [], 0             
                print("ko doi mu bao hiem")
            elif flag_red_light and 1 in list_vehicle_pass_line_mode:    
                phat_loa_nhac_nho_dung_dung_vach()
            # if not flag_red_light:
            #     list_index_plate_excel_detected, list_wear_helmet = [], []
            #     count_vehicle_pass_on_line = 0
        #print(list_wear_helmet)
        # print("Time: ", time.time() - start)
        # print("FPS: {}".format(1/(time.time() - start)))
        cv2.imwrite("E:\\project\\result\\{}".format(filename), image_ori)
        cv2.imshow("", cv2.resize(image_ori, (1040, 780))) 
        if first_frame and -0.1 < slope < 0.1 and intercept != 0:
            first_frame = False
        if cv2.waitKey(0) == 27:
            break
  
    cv2.destroyAllWindows()

