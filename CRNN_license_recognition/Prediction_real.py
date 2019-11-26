import cv2
import itertools, os, time
import numpy as np
# from Model import get_Model
from Model_GRU import get_Model
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
    char_remain = label[2:]

    try:
        char1 = char1 if char1 != 'Z' else ''
        char2 = char2 if char2 != 'Z' else ''
    except:
        pass

    return char1 + char2 + char_remain


parser = argparse.ArgumentParser()
parser.add_argument("-w", "--weight", help="weight file directory",
                    type=str, default="/home/truongdongdo/Desktop/CRNN-Keras/checkpoint/CRNN_GRU_23_11_real_remake.hdf5")
parser.add_argument("-t", "--test_img", help="Test image directory",
                    # type=str, default="/home/truongdongdo/Desktop/CRNN-Keras/DB/type5/test_1/")
                    type=str, default="/home/truongdongdo/Desktop/CRNN-Keras/test_real/test_only/")
args = parser.parse_args()

# Get CRNN model
model = get_Model(training=False)

try:
    model.load_weights(args.weight)
    print("...Previous weight data...")
except:
    raise Exception("No weight file!")


test_dir =args.test_img
test_imgs = os.listdir(args.test_img)
total = 0
acc = 0
start = time.time()
for test_img in test_imgs:
    letter_total = 0
    letter_acc = 0

    img = cv2.imread(test_dir + test_img, cv2.IMREAD_GRAYSCALE)

    img_pred = img.astype(np.float32)
    img_pred = cv2.resize(img_pred, (128, 64))
    img_pred = (img_pred / 255.0) * 2.0 - 1.0
    img_pred = img_pred.T
    img_pred = np.expand_dims(img_pred, axis=-1)
    img_pred = np.expand_dims(img_pred, axis=0)
    
    net_out_value = model.predict(img_pred)

    pred_texts = decode_label(net_out_value)

    # for i in range(min(len(pred_texts), len(test_img[0:-4]))):
    #     if pred_texts[i] == test_img[i]:
    #         letter_acc += 1
    # letter_total += max(len(pred_texts), len(test_img[0:-4]))
    # print("letter ACC : ", letter_acc / letter_total)

    # pred_texts = pred_texts[-5:]
    # true_label = test_img[0:-4][-5:]
    pred_texts = pred_texts
    true_label = test_img[0:-4]
    for i in range(min(len(pred_texts), len(true_label))):
        if pred_texts[i] == true_label[i]:
            letter_acc += 1
    letter_total = max(len(pred_texts), len(true_label))
    # print(letter_total)
    
    if pred_texts == true_label:
        acc += 1
    total += 1
    print('Predicted: %s  /  True: %s' % (remove_ZZ(pred_texts), true_label))
    print("letter ACC : ", letter_acc / letter_total)
    
    img = cv2.resize(img, (128, 64))
    # cv2.rectangle(img, (0,0), (150, 30), (0,0,0), -1)
    # cv2.putText(img, pred_texts, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)

    cv2.imshow(pred_texts, img)

end = time.time()
total_time = (end - start)
print("Time : ",total_time / total)
print("ACC : ", acc / total)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
