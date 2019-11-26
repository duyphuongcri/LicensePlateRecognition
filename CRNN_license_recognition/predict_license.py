import cv2
import itertools, os, time
import numpy as np
from Model import get_Model
from parameter import letters
import argparse
from keras import backend as K
K.set_learning_phase(0)

def decode_label(out):
    # out : (1, 32, 42)
    out_best = list(np.argmax(out[0, 2:], axis=1))  # get max index -> len = 32
    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
    outstr = ''
    for i in out_best:
        if i < len(letters):
            outstr += letters[i]
    return outstr

def remove_ZZ(label): 
    char1 = label[0]
    char2 = label[1]
    char_remain = label[1:]

    try:
        char1 = char1 if char1 != 'Z' else ''
        char2 = char2 if char2 != 'Z' else ''
    except:
        pass

    return char1 + char2 + char_remain

def predict_license(img):
    start = time.time()

    img = img.astype(np.float32)
    img = cv2.resize(img, (128, 64))
    img_pred = (img / 255.0) * 2.0 - 1.0
    img_pred = img_pred.T
    img_pred = np.expand_dims(img_pred, axis=-1)
    img_pred = np.expand_dims(img_pred, axis=0)
    
    net_out_value = model.predict(img_pred)

    pred_texts = decode_label(net_out_value)
    license_num = remove_ZZ(pred_texts)

    # print('Predicted: %s  ' % license_num)
    
    # cv2.imshow(pred_texts, img)

    end = time.time()
    total_time = (end - start)
    print("Time : ",total_time)
    return license_num

if __name__ == '__main__':
    # Get CRNN model
    model_license_path = "/home/truongdongdo/Desktop/CRNN-Keras/checkpoint/LSTM+BN5_11_23_aug.hdf5"
    model = get_Model(training=False)
    try:
        model.load_weights(model_license_path)
        print("...Previous weight data...")
    except:
        raise Exception("No weight file!")

    # image from detector
    test_img_dir = "/home/truongdongdo/Desktop/CRNN-Keras/test_real/test_only/71B321844.png"
    img = cv2.imread(test_img_dir, cv2.IMREAD_GRAYSCALE)

    license_num = predict_license(img)
    print('Predicted: %s  ' % license_num)