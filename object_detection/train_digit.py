import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2
import keras
import glob
import os
from time import time
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import pickle
from keras.callbacks import TensorBoard

cwd = os.getcwd()
logsdir = os.path.join(cwd,"logs")
batchsize = 8
number_class = 10
epochs = 50
ratio_testset = 0.15
lrate = 0.001


y0 = len(glob.glob(os.path.join("Dataset\\0" ,'*png')))
y1 = len(glob.glob(os.path.join("Dataset\\1" ,'*png')))
y2 = len(glob.glob(os.path.join("Dataset\\2" ,'*png')))
y3 = len(glob.glob(os.path.join("Dataset\\3" ,'*png')))
y4 = len(glob.glob(os.path.join("Dataset\\4" ,'*png')))
y5 = len(glob.glob(os.path.join("Dataset\\5" ,'*png')))
y6 = len(glob.glob(os.path.join("Dataset\\6" ,'*png')))
y7 = len(glob.glob(os.path.join("Dataset\\7" ,'*png')))
y8 = len(glob.glob(os.path.join("Dataset\\8" ,'*png')))
y9 = len(glob.glob(os.path.join("Dataset\\9" ,'*png')))

# X_data
X = []
flag = 0
for i in glob.glob(os.path.join("Dataset\\0" ,'*png')): #["C:\a.png", "C:b.png"]
	img = cv2.imread(i,0)
	if flag == 0:
		X = img 
		flag = 1
	else:
		X = np.append(X,img, axis = 0)	
print("Loading 10%..")
for i in glob.glob(os.path.join("Dataset\\1" ,'*png')): 
	img = cv2.imread(i,0)
	X = np.append(X,img, axis = 0)
print("Loading 20%..")
for i in glob.glob(os.path.join("Dataset\\2" ,'*png')): #120
	img = cv2.imread(i,0)
	X = np.append(X,img, axis = 0)
print("Loading 30%..")
for i in glob.glob(os.path.join("Dataset\\3" ,'*png')): #130
	img = cv2.imread(i,0)	
	X = np.append(X,img, axis = 0)
print("Loading 40%..")
for i in glob.glob(os.path.join("Dataset\\4" ,'*png')): #140
	img = cv2.imread(i,0)
	X = np.append(X,img, axis = 0)
print("Loading 50%..")
for i in glob.glob(os.path.join("Dataset\\5" ,'*png')): #150
	img = cv2.imread(i,0)
	X = np.append(X,img, axis = 0)
print("Loading 60%..")
for i in glob.glob(os.path.join("Dataset\\6" ,'*png')): #160
	img = cv2.imread(i,0)
	X = np.append(X,img, axis = 0)
print("Loading 70%..")
for i in glob.glob(os.path.join("Dataset\\7" ,'*png')): #160
	img = cv2.imread(i,0)	
	X = np.append(X,img, axis = 0)
print("Loading 80%..")
for i in glob.glob(os.path.join("Dataset\\8" ,'*png')): #160
	img = cv2.imread(i,0)	
	X = np.append(X,img, axis = 0)
print("Loading 90%..")
for i in glob.glob(os.path.join("Dataset\\9" ,'*png')): #160
	img = cv2.imread(i,0)	
	X = np.append(X,img, axis = 0)
print("Finished")

X = X.reshape(-1,64,64,1)
X.astype('float32')
X = X / 255
#y data
y = np.asarray([0]*y0 + [1]*y1 + [2]*y2 + [3]*y3 + [4]*y4 + [5]*y5 + [6]*y6 + [7]*y7 + [8]*y8 + [9]*y9)
y.astype('uint8')

########################
model = Sequential()
model.add(Conv2D(16, kernel_size=(5, 5),				 
                 activation='relu',
				 strides=(1, 1),
                 input_shape= (64,64,1),
				 padding='same',)) # valid
model.add(MaxPooling2D(pool_size=(2, 2), 
                        strides=None, 
                        padding='valid', 
                        data_format=None))
model.add(Conv2D(32, kernel_size=(5, 5), 
				 activation='relu',
				 strides=(2, 2),
				 padding='valid',))
model.add(MaxPooling2D(pool_size=(2, 2), 
                        strides=None, 
                        padding='valid', 
                        data_format=None))
model.add(Conv2D(64, kernel_size=(5, 5), 
                    activation='relu',
                    strides=(1, 1),
                    padding='valid',))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(number_class, activation='softmax'))
model.summary()
###########################################


#--------------------Train/Test-----------------------
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=ratio_testset, random_state=1)
y_train = np_utils.to_categorical(y_train, number_class)
y_valid = np_utils.to_categorical(y_valid, number_class)

#--------------------Train with splitting data follows as Train/Test method------------
checkpoint_path = os.path.join(logsdir, "{epoch:04d}.h5")
tensorboard = [TensorBoard(log_dir=logsdir,
							histogram_freq=0, write_graph=True, write_images=False),
							keras.callbacks.ModelCheckpoint(checkpoint_path,
							verbose=0, save_best_only=True),]
Adam = keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['accuracy'])

model.fit(X_train, y_train,
					batch_size=batchsize,
					epochs=epochs,
					shuffle=True,
					validation_data=(X_valid, y_valid),
					callbacks=tensorboard)