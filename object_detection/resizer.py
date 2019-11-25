## Bulk image resizer

# This script simply resizes all the images in a folder to one-eigth their
# original size. It's useful for shrinking large cell phone pictures down
# to a size that's more manageable for model training.

# Usage: place this script in a folder of images you want to shrink,
# and then run it.

import numpy as np
import cv2
import os

path = "E:\\project\\LicensePLate"
files = [i for i in os.listdir(path) if i.endswith(".png")]

for filename in files:

    image = cv2.imread(os.path.join(path, filename))
    h, w = image.shape[:2]
    
    pd = w - h
    if pd > 0:
        padding = [(0, pd), (0, 0),(0,0)]
    else:
        padding = [(0, 0), (0, -pd),(0,0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    resized = cv2.resize(image,(300, 300), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join("E:\\project\\LicensePLate_resized", filename),resized)
