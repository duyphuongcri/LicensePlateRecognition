import numpy as np 
import cv2
import keras
from keras.models import load_model

def fit_size(img):
    h, w = img.shape[:2]
    print("Shape of Image:", h, w)
    pd = int((h - w)/2)
    padding = [(0, 0), (pd, pd)]
    img_padded = np.pad(img, padding, mode='constant', constant_values=0)
    output = cv2.resize(img_padded, (64,64), interpolation=cv2.INTER_AREA)
    return output

model = load_model('E:\\Tensorflow-API\\object_detection\\logs\\0002.h5')

img = cv2.imread("E:\\project\\letter\\image001833.png", 0)
img = fit_size(img)

img = img.reshape(1,64,64,1)
img = img / 255
img_class = model.predict(img)
print(img_class.argmax())