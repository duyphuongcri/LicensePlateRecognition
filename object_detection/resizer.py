## Bulk image resizer

# This script simply resizes all the images in a folder to one-eigth their
# original size. It's useful for shrinking large cell phone pictures down
# to a size that's more manageable for model training.

# Usage: place this script in a folder of images you want to shrink,
# and then run it.

import numpy as np
import cv2
import os

path = "E:\\Tensorflow-API\\object_detection\\Dataset\\10"
files = [i for i in os.listdir(path) if i.endswith(".png")]

for filename in files:

    image = cv2.imread(os.path.join(path, filename), 0)
    h, w = image.shape[:2]
    
    pd = int((h - w)/2)
    padding = [(0, 0), (pd, pd)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    resized = cv2.resize(image,(64, 64), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(path, filename),resized)
