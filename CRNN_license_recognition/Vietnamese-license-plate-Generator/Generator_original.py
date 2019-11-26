import os, random
import cv2, argparse
import numpy as np
import Automold as am
import Helpers as hp
import math

def random_bright(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.array(img, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img

def image_augmentation_aug(img, ang_range=6, shear_range=3, trans_range=2):
    # Rotation
    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    rows, cols, ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 0.9)

    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    shear_M = cv2.getAffineTransform(pts1, pts2)

    img = cv2.warpAffine(img, Rot_M, (cols, rows))
    img = cv2.warpAffine(img, Trans_M, (cols, rows))
    img = cv2.warpAffine(img, shear_M, (cols, rows))

    # Brightness
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.array(img, dtype=np.float64)
    random_bright = .2 + np.random.uniform(0,0.8)
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    # Blur
    blur_value = random.randint(0,8) * 2 + 1
    img = cv2.blur(img,(blur_value, blur_value))

    # Add shadow
    img= am.add_shadow(img)

    # add_sun_flare
    img= am.add_sun_flare(img, flare_center=(10,10),angle=-math.pi/3)

    return img

class ImageGenerator:
    def __init__(self, save_path):
        self.save_path = save_path
        # Plate
        self.plate = cv2.imread("plate.jpg")

        # loading Number
        file_path = "./num/"
        file_list = os.listdir(file_path)
        self.Number = list()
        self.number_list = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Number.append(img)
            self.number_list.append(file[0:-4])

        # loading Char
        file_path = "./char/"
        file_list = os.listdir(file_path)
        self.char_list = list()
        self.Char1 = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Char1.append(img)
            self.char_list.append(file[0:-4])

        #=========================================================================


    def Type_1(self, num, save=False):
        number = [cv2.resize(number, (56, 83)) for number in self.Number]
        char = [cv2.resize(char1, (56, 83)) for char1 in self.Char1]
        Plate = cv2.resize(self.plate, (520, 110))

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (520, 110))
            label = "Z"
            # row -> y , col -> x
            row, col = 13, 21  # row + 83, col + 56
            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # character 3
            rand_int = random.randint(0, 25)
            label += self.char_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = char[rand_int]
            col += (56 + 30)

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 8
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # Plate = random_bright(Plate)
            Plate = image_augmentation_aug(Plate)
            if save:
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def Type_6(self, num, save=False):
            number = [cv2.resize(number, (56, 83)) for number in self.Number]
            char = [cv2.resize(char1, (56, 83)) for char1 in self.Char1]
            Plate = cv2.resize(self.plate, (464, 110))

            for i, Iter in enumerate(range(num)):
                Plate = cv2.resize(self.plate, (464, 110))
                label = "ZZ"
                # row -> y , col -> x
                row, col = 13, 21  # row + 83, col + 56
                # number 1
                rand_int = random.randint(0, 9)
                label += self.number_list[rand_int]
                Plate[row:row + 83, col:col + 56, :] = number[rand_int]
                col += 56

                # number 2
                rand_int = random.randint(0, 9)
                label += self.number_list[rand_int]
                Plate[row:row + 83, col:col + 56, :] = number[rand_int]
                col += 56

                # character 3
                rand_int = random.randint(0, 25)
                label += self.char_list[rand_int]
                Plate[row:row + 83, col:col + 56, :] = char[rand_int]
                col += (56 + 30)

                # number 4
                rand_int = random.randint(0, 9)
                label += self.number_list[rand_int]
                Plate[row:row + 83, col:col + 56, :] = number[rand_int]
                col += 56

                # number 5
                rand_int = random.randint(0, 9)
                label += self.number_list[rand_int]
                Plate[row:row + 83, col:col + 56, :] = number[rand_int]
                col += 56

                # number 6
                rand_int = random.randint(0, 9)
                label += self.number_list[rand_int]
                Plate[row:row + 83, col:col + 56, :] = number[rand_int]
                col += 56

                # number 7
                rand_int = random.randint(0, 9)
                label += self.number_list[rand_int]
                Plate[row:row + 83, col:col + 56, :] = number[rand_int]
                col += 56

                # Plate = random_bright(Plate)
                Plate = image_augmentation_aug(Plate)
                if save:
                    cv2.imwrite(self.save_path + label + ".jpg", Plate)
                else:
                    cv2.imshow(label, Plate)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

    
    def Type_2(self, num, save=False):
        number1 = [cv2.resize(number, (56, 80)) for number in self.Number]
        char = [cv2.resize(char1, (56, 80)) for char1 in self.Char1]

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (269, 190))

            label = "Z"
            # row -> y , col -> x
            row, col = 10, 13

            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 56, :] = number1[rand_int]
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 56, :] = number1[rand_int]
            col += 56 + 20
            
            # character 3
            rand_int = random.randint(0, 25)
            label += self.char_list[rand_int]
            Plate[row:row + 80, col:col + 56, :] = char[rand_int]
            col += 56

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 56, :] = number1[rand_int]           


            row, col = 95, 13

           # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 56, :] = number1[rand_int]
            col += 56 + 7

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 56, :] = number1[rand_int]
            col += 56 + 7

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 56, :] = number1[rand_int]
            col += 56 + 7 

            # number 8
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 56, :] = number1[rand_int]
            col += 56

            # Plate = random_bright(Plate)
            Plate = image_augmentation_aug(Plate)
            if save:
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def Type_3(self, num, save=False):
        number1 = [cv2.resize(number, (56, 80)) for number in self.Number]
        char = [cv2.resize(char1, (56, 80)) for char1 in self.Char1]

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (269, 190))

            label = "Z"
            # row -> y , col -> x
            row, col = 10, 13

            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 56, :] = number1[rand_int]
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 56, :] = number1[rand_int]
            col += 56 + 20
            
            # character 3
            rand_int = random.randint(0, 25)
            label += self.char_list[rand_int]
            Plate[row:row + 80, col:col + 56, :] = char[rand_int]
            col += 56

            # character 4
            rand_int = random.randint(0, 25)
            label += self.char_list[rand_int]
            Plate[row:row + 80, col:col + 56, :] = char[rand_int]
            col += 56        

            row, col = 95, 13

           # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 56, :] = number1[rand_int]
            col += 56 + 7

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 56, :] = number1[rand_int]
            col += 56 + 7

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 56, :] = number1[rand_int]
            col += 56 + 7 

            # number 8
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 56, :] = number1[rand_int]
            col += 56

            # Plate = random_bright(Plate)
            Plate = image_augmentation_aug(Plate)
            if save:
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def Type_4(self, num, save=False):
        number1 = [cv2.resize(number, (56, 80)) for number in self.Number]
        number2 = [cv2.resize(number, (45, 80)) for number in self.Number]
        char = [cv2.resize(char1, (56, 80)) for char1 in self.Char1]

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (269, 190))

            label = "Z"
            # row -> y , col -> x
            row, col = 10, 50

            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 56, :] = number1[rand_int]
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 56, :] = number1[rand_int]
            col += 56
            
            # character 3
            rand_int = random.randint(0, 25)
            label += self.char_list[rand_int]
            Plate[row:row + 80, col:col + 56, :] = char[rand_int]
            col += 56

            row, col = 95, 10

           # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 45, :] = number2[rand_int]
            col += 45

           # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 45, :] = number2[rand_int]
            col += 45
            
            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 45, :] = number2[rand_int]
            col += 45 + 20 

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 45, :] = number2[rand_int]
            col += 45

            # number 8
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 45, :] = number2[rand_int]
            col += 45            


            # Plate = random_bright(Plate)
            Plate = image_augmentation_aug(Plate)
            if save:
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def Type_5(self, num, save=False):
        number1 = [cv2.resize(number, (45, 80)) for number in self.Number]
        char = [cv2.resize(char1, (45, 80)) for char1 in self.Char1]

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (269, 190))

            label = str()
            # row -> y , col -> x
            row, col = 10, 20

            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 45, :] = number1[rand_int]
            col += 45

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 45, :] = number1[rand_int]
            col += 45 + 49
            
            # character 3
            rand_int = random.randint(0, 25)
            label += self.char_list[rand_int]
            Plate[row:row + 80, col:col + 45, :] = char[rand_int]
            col += 45

            # character 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 45, :] = number1[rand_int]
            col += 45

            row, col = 95, 10

           # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 45, :] = number1[rand_int]
            col += 45

           # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 45, :] = number1[rand_int]
            col += 45
            
            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 45, :] = number1[rand_int]
            col += 45 + 20 

            # number 8
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 45, :] = number1[rand_int]
            col += 45

            # number 9
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 80, col:col + 45, :] = number1[rand_int]
            col += 45            


            # Plate = random_bright(Plate)
            Plate = image_augmentation_aug(Plate)
            if save:
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--img_dir", help="save image directory",
                    type=str, default="/home/truongdongdo/Desktop/CRNN-Keras/DB/type5/test/")
parser.add_argument("-n", "--num", help="number of image",
                    type=int)
parser.add_argument("-s", "--save", help="save or imshow",
                    type=bool, default=True)
args = parser.parse_args()


img_dir = args.img_dir
A = ImageGenerator(img_dir)

num_img = args.num
Save = args.save

# A.Type_1(num_img, save=Save)
# print("Type 1 finish")
# A.Type_2(num_img, save=Save)
# print("Type 2 finish")
# A.Type_3(num_img, save=Save)
# print("Type 3 finish")
# A.Type_4(num_img, save=Save)
# print("Type 4 finish")
A.Type_5(num_img, save=Save)
print("Type 5 finish")
# A.Type_6(num_img, save=Save)
# print("Type 6 finish")
