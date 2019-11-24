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
PATH_TO_CKPT_3classes = os.path.join(CWD_PATH,"Car",'frozen_inference_graph.pb')
PATH_TO_CKPT_LicensePlate = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS_3classes = os.path.join(CWD_PATH,'training','labelmap_3classes.pbtxt')
PATH_TO_LABELS_LicensePlate = os.path.join(CWD_PATH,'training','labelmap.pbtxt')


# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map_3classes = label_map_util.load_labelmap(PATH_TO_LABELS_3classes)
categories_3classes = label_map_util.convert_label_map_to_categories(label_map_3classes, max_num_classes=3, use_display_name=True)
category_index_3classes = label_map_util.create_category_index(categories_3classes)

label_map_LicensePlate = label_map_util.load_labelmap(PATH_TO_LABELS_LicensePlate)
categories_LicensePlate = label_map_util.convert_label_map_to_categories(label_map_LicensePlate, max_num_classes=1, use_display_name=True)
category_index_LicensePlate = label_map_util.create_category_index(categories_LicensePlate)
print(categories_LicensePlate)

# Load the Tensorflow model into memory.
detection_graph_3classes = tf.Graph()
with detection_graph_3classes.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT_3classes, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess_3classes = tf.Session(graph=detection_graph_3classes)

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
image_tensor_3classes = detection_graph_3classes.get_tensor_by_name('image_tensor:0')
image_tensor_LicensePlate = detection_graph_LicensePlate.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes_3classes = detection_graph_3classes.get_tensor_by_name('detection_boxes:0')
detection_boxes_LicensePlate = detection_graph_LicensePlate.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores_3classes = detection_graph_3classes.get_tensor_by_name('detection_scores:0')
detection_scores_LicensePlate = detection_graph_LicensePlate.get_tensor_by_name('detection_classes:0')

detection_classes_3classes = detection_graph_3classes.get_tensor_by_name('detection_classes:0')
detection_classes_LicensePlate  = detection_graph_LicensePlate .get_tensor_by_name('detection_classes:0')
# Number of objects detected
num_detections_3classes = detection_graph_3classes.get_tensor_by_name('num_detections:0')
num_detections_LicensePlate = detection_graph_LicensePlate.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
import time
path = "E:\\project\\data"
files = [i for i in os.listdir(path) if i.endswith(".png")]
n=0
for filename in files:
    start = time.time()
    print(filename)
    image_ori = cv2.imread(os.path.join(path, filename))

    h_ori, w_ori = image_ori.shape[:2]
    #print("size of original Image", h_ori, w_ori)



    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess_3classes.run(
        [detection_boxes_3classes, detection_scores_3classes, detection_classes_3classes, num_detections_3classes],
        feed_dict={image_tensor_3classes: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')

    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     image,
    #     np.squeeze(boxes),
    #     np.squeeze(classes).astype(np.int32),
    #     np.squeeze(scores),
    #     category_index_3classes,
    #     use_normalized_coordinates=True,
    #     line_thickness=1,
    #     min_score_thresh=0.20)
   
    for i in range(10):
        if scores[0][i] > 0.5:
            y1, x1, y2, x2 = int(boxes[0][i][0]*image.shape[0]), int(boxes[0][i][1]*image.shape[1]), int(boxes[0][i][2]*image.shape[0]), int(boxes[0][i][3]*image.shape[1])
            #cv2.rectangle(image, (x1,y1),(x2,y2), (0,255,0), 2)
            y1_vehicle_ori, x1_vehicle_ori, y2_vehicle_ori, x2_vehicle_ori = int(y1*2.25), int(x1*2.25), int(y2*2.25), int(x2*2.25)
            #print("Size of vehicle: {}x{}".format(y2_vehicle_ori - y1_vehicle_ori, x2_vehicle_ori - x1_vehicle_ori))
            w_box = x2_vehicle_ori - x1_vehicle_ori
            w_region = int(w_box*0.9)
            x1_region = int(x1_vehicle_ori + w_box*0.05)
            mode = False
            if y2_vehicle_ori >  w_region:
                img_region_plate = image_ori[y2_vehicle_ori - w_region:y2_vehicle_ori, x1_region: x1_region + w_region]
                mode = False
            else:
                mode = True
                img_region_plate = np.ones((w_region, w_region, 3), dtype=np.uint8)*image_ori[y2_vehicle_ori][x1_vehicle_ori]
                img_region_plate[0: y2_vehicle_ori - y1_vehicle_ori, 0:w_region] = image_ori[y1_vehicle_ori:y2_vehicle_ori,  x1_region: x1_region + w_region]
            #print("Size of region: {}x{}".format(img_region_plate.shape[0], img_region_plate.shape[1]))
            img_region_plate = cv2.resize(img_region_plate,(300,300))

            img_region_expanded = np.expand_dims(img_region_plate, axis=0)

            # Perform the actual detection by running the model with the image as input
            (boxes_LicensePlate, scores_LicensePlate, classes_LicensePlate, num_LicensePlate) = sess_LicensePlate.run(
                [detection_boxes_LicensePlate, detection_scores_LicensePlate, detection_classes_LicensePlate, num_detections_LicensePlate],
                feed_dict={image_tensor_LicensePlate: img_region_expanded})
            for j in range(1):
                if scores_LicensePlate[0][j] > 0.8:
                    #print(scores_LicensePlate[0][j])
                    y1_crop, x1_crop, y2_crop, x2_crop = int(boxes_LicensePlate[0][j][0]*img_region_plate.shape[0]), int(boxes_LicensePlate[0][j][1]*img_region_plate.shape[1]), int(boxes_LicensePlate[0][i][2]*img_region_plate.shape[0]), int(boxes_LicensePlate[0][j][3]*img_region_plate.shape[1])
                    ratio = w_region / img_region_plate.shape[0] 
                    print(ratio, w_region, img_region_plate.shape[0])
                    
                    y1_Plate_ori, x1_Plate_ori, y2_Plate_ori, x2_Plate_ori = int(y1_crop*ratio + y2_vehicle_ori - w_region), int(x1_crop*ratio + x1_region), int(y2_crop*ratio + y2_vehicle_ori - w_region), int(x2_crop*ratio + x1_region)
            

                    cv2.rectangle(image_ori, (x1_Plate_ori, y1_Plate_ori),(x2_Plate_ori, y2_Plate_ori), (0,0,255), 2)
                    cv2.putText(image_ori,'Plate-{:.02f}'.format(scores_LicensePlate[0][j]), (x1_Plate_ori, y1_Plate_ori), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),3)

                    cv2.rectangle(img_region_plate, (x1_crop, y1_crop),(x2_crop, y2_crop), (0,255,0), 2)
                    cv2.imshow("a",img_region_plate )
                else:
                    break


            # cv2.imwrite("E:\\project\\data_LicensePlate\\image%03d.png" %n, img_region_plate)
            # n += 1
            # cv2.imshow("iamge", img_region_plate)
            # cv2.waitKey(0)

            cv2.rectangle(image_ori, (x1_vehicle_ori, y1_vehicle_ori),(x2_vehicle_ori, y2_vehicle_ori), (0,255,0), 2)
            cv2.putText(image_ori,'Vehicle', (x1_vehicle_ori, y1_vehicle_ori), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),3)
        else:
            break

    # All the results have been drawn on image. Now display the image.
    cv2.imshow("", image_ori) 
    if cv2.waitKey(0) == 27:
        break
    print("timing: ",time.time() - start)
cv2.destroyAllWindows()