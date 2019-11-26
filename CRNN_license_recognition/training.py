from keras import backend as K
from keras.optimizers import Adadelta,Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from Image_Generator import TextImageGenerator
from Model_GRU import get_Model
from parameter import *
K.set_learning_phase(0)

# # Model description and training

model = get_Model(training=True)

try:
    model.load_weights('/home/truongdongdo/Desktop/CRNN-Keras/checkpoint/CRNN_GRU_23_11_type5_new.hdf5')
    print("...Previous weight data...")
except:
    print("...New weight data...")
    pass

train_file_path = '/home/truongdongdo/Desktop/CRNN-Keras/DB/type5/train/'
tiger_train = TextImageGenerator(train_file_path, img_w, img_h, batch_size, downsample_factor)
tiger_train.build_data()

valid_file_path = '/home/truongdongdo/Desktop/CRNN-Keras/DB/type5/test/'
tiger_val = TextImageGenerator(valid_file_path, img_w, img_h, val_batch_size, downsample_factor)
tiger_val.build_data()

# ada = Adadelta()
ada = Adam()

save_model_path = "/home/truongdongdo/Desktop/CRNN-Keras/checkpoint/"

early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=20, mode='min', verbose=1)
# checkpoint = ModelCheckpoint(filepath= save_model_path + 'LSTM+BN5--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='loss', verbose=1, mode='min', period=1)
checkpoint = ModelCheckpoint(filepath= save_model_path + 'CRNN_GRU_23_11_type5_new_1.hdf5', monitor='loss', verbose=1, mode='min', period=1)
# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)

# captures output of softmax so we can decode the output during visualization
model.fit_generator(generator=tiger_train.next_batch(),
                    steps_per_epoch=int(tiger_train.n / batch_size),
                    epochs=10000,
                    callbacks=[checkpoint,early_stop],
                    validation_data=tiger_val.next_batch(),
                    validation_steps=int(tiger_val.n / val_batch_size), verbose=1)
