######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained neural network to perform object detection.
# It loads the classifier and uses it to perform object detection on an image.
# It draws boxes, scores, and labels around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
from __future__ import print_function
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import matplotlib
from matplotlib import pyplot as plt
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'test\\image000.jpg'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')


# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

detection_classes  = detection_graph.get_tensor_by_name('detection_classes:0')
# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value


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
    for i in range(5):
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
        img_region_plate = cv2.resize(img_region_plate, (300, 300))
        #cv2.imwrite("E:\\project\\data_helmet\\{0}_{1}.png".format(filename[:-4], i), img_region_plate)

        list_region_plate.append(img_region_plate)   
    return list_region_plate

import time
path = "E:\\project\\data"
files = [i for i in os.listdir(path) if i.endswith(".png") and i.startswith("image")]
n=0
for filename in files:
    print(filename)
    image_ori = cv2.imread(os.path.join(path, filename))
    #image_ori = cv2.imread("E:\\Capture.PNG")
    #image_ori = image_ori[:, 240:1680]
    img_resized = cv2.resize(image_ori, (640, 480))
    # vehicle_boxes, classID = detect_vehicle(image_ori, 
    #                                         sess, 
    #                                         detection_boxes, 
    #                                         detection_scores, 
    #                                         detection_classes, 
    #                                         num_detections, 
    #                                         image_tensor,
    #                                         threshold_score=0.8)
    # _ = crop_vehicle_region(image_ori, vehicle_boxes, classID)                                        
    # # h_ori, w_ori = image_ori.shape[:2]

    #print("size of original Image", h_ori, w_ori)
    image_expanded = np.expand_dims(img_resized, axis=0)
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visualize the results')
    
    vis_util.visualize_boxes_and_labels_on_image_array(
        img_resized,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=1,
        min_score_thresh=0.8)
    # boxes[0][:,0] = boxes[0][:,0]*1080
    # boxes[0][:,1] = boxes[0][:,1]*1440
    # boxes[0][:,2] = boxes[0][:,2]*1080
    # boxes[0][:,3] = boxes[0][:,3]*1440
    # index = non_max_suppression(boxes[0][:int(num[0])],scores[0][:int(num[0])], 0.3)
    # n = 0
    # for i in range(len(boxes[0])):
    #     y1, x1, y2, x2 = boxes[0][i][:]
    #     h, w = y2 - y1, x2 - x1
    #     if scores[0][i] > 0.5 and w / h < 2:
    #         print("shape:", int(h/2), int(w))
    #         # if y1 < image_ori.shape[0]/2 and y2 > image_ori.shape[0]/2 and classes[0][i] == 2:
    #         #     cv2.imwrite("E:\\project\\data_plate\\{0}_{1}.png".format(filename[:-4], i), image_ori[int(y1 + h/2): int(y2), int(x1): int(x2)])   
    #         cv2.rectangle(image_ori, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    #         cv2.putText(image_ori, "{}".format(int(classes[0][i])), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),2)    
    #     else: 
    #         break
    # All the results have been drawn on image. Now display the image.
    cv2.imshow("", img_resized) 
    if cv2.waitKey(0) == 27:
        break
cv2.destroyAllWindows()