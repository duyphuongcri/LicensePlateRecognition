import os, random
import cv2, argparse
import numpy as np

def image_augmentation(img, type2=False):
    # perspective
    w, h, _ = img.shape
    pts1 = np.float32([[0, 0], [0, w], [h, 0], [h, w]])
    # 좌표의 이동점
    begin, end = 30, 90
    pts2 = np.float32([[random.randint(begin, end), random.randint(begin, end)],
                       [random.randint(begin, end), w - random.randint(begin, end)],
                       [h - random.randint(begin, end), random.randint(begin, end)],
                       [h - random.randint(begin, end), w - random.randint(begin, end)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img = cv2.warpPerspective(img, M, (h, w))

    # Brightness
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.array(img, dtype=np.float64)
    random_bright = .4 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    # Blur
    blur_value = random.randint(0,4) * 2 + 1
    img = cv2.blur(img,(blur_value, blur_value))
    if type2:
        return img[130:280, 180:600, :]
    return img[130:280, 120:660, :]


class ImageGenerator:
    def __init__(self, save_path):
        self.save_path = save_path
        # Plate
        self.plate = cv2.imread("plate.jpg")

        # loading Number ====================  white-one-line  ==========================
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

    def Type_1(self, num, save=False):
        number = [cv2.resize(number, (56, 83)) for number in self.Number]
        char = [cv2.resize(char1, (56, 83)) for char1 in self.Char1]

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (520, 110))
            b_width ,b_height = 400, 800
            random_R, random_G, random_B = random.randint(0,255), random.randint(0,255), random.randint(0,255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width), (random_R, random_G, random_B), -1)

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

            s_width, s_height = int((400-110)/2), int((800-520)/2)
            background[s_width:110 + s_width, s_height:520 + s_height, :] = Plate
            background = image_augmentation(background)

            if save:
                cv2.imwrite(self.save_path + label + ".jpg", background)
            else:
                cv2.imshow(label, background)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def Type_6(self, num, save=False):
        number = [cv2.resize(number, (56, 83)) for number in self.Number]
        char = [cv2.resize(char1, (56, 83)) for char1 in self.Char1]

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (520, 110))
            b_width ,b_height = 400, 800
            random_R, random_G, random_B = random.randint(0,255), random.randint(0,255), random.randint(0,255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width), (random_R, random_G, random_B), -1)

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

            # number 8
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            s_width, s_height = int((400-110)/2), int((800-520)/2)
            background[s_width:110 + s_width, s_height:520 + s_height, :] = Plate
            background = image_augmentation(background)

            if save:
                cv2.imwrite(self.save_path + label + ".jpg", background)
            else:
                cv2.imshow(label, background)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def Type_2(self, num, save=False):
        number1 = [cv2.resize(number, (56, 80)) for number in self.Number]
        char = [cv2.resize(char1, (56, 80)) for char1 in self.Char1]

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (269, 190))
            b_width ,b_height = 400, 800
            random_R, random_G, random_B = random.randint(0,255), random.randint(0,255), random.randint(0,255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width), (random_R, random_G, random_B), -1)        

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

            s_width, s_height = int((400-190)/2), int((800-269)/2)
            background[s_width:190 + s_width, s_height:269 + s_height, :] = Plate
            background = image_augmentation(background)


            if save:
                cv2.imwrite(self.save_path + label + ".jpg", background)
            else:
                cv2.imshow(label, background)
                cv2.waitKey(0)
                cv2.destroyAllWindows()    

    def Type_3(self, num, save=False):
        number1 = [cv2.resize(number, (56, 80)) for number in self.Number]
        char = [cv2.resize(char1, (56, 80)) for char1 in self.Char1]

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (269, 190))
            b_width ,b_height = 400, 800
            random_R, random_G, random_B = random.randint(0,255), random.randint(0,255), random.randint(0,255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width), (random_R, random_G, random_B), -1)        
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

            s_width, s_height = int((400-190)/2), int((800-269)/2)
            background[s_width:190 + s_width, s_height:269 + s_height, :] = Plate
            background = image_augmentation(background)

            if save:
                cv2.imwrite(self.save_path + label + ".jpg", background)
            else:
                cv2.imshow(label, background)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def Type_4(self, num, save=False):
        number1 = [cv2.resize(number, (56, 80)) for number in self.Number]
        number2 = [cv2.resize(number, (45, 80)) for number in self.Number]
        char = [cv2.resize(char1, (56, 80)) for char1 in self.Char1]

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (269, 190))
            b_width ,b_height = 400, 800
            random_R, random_G, random_B = random.randint(0,255), random.randint(0,255), random.randint(0,255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width), (random_R, random_G, random_B), -1)

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


            s_width, s_height = int((400-190)/2), int((800-269)/2)
            background[s_width:190 + s_width, s_height:269 + s_height, :] = Plate
            background = image_augmentation(background)

            if save:
                cv2.imwrite(self.save_path + label + ".jpg", background)
            else:
                cv2.imshow(label, background)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def Type_5(self, num, save=False):
        number1 = [cv2.resize(number, (45, 80)) for number in self.Number]
        char = [cv2.resize(char1, (45, 80)) for char1 in self.Char1]

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (269, 190))
            b_width ,b_height = 400, 800
            random_R, random_G, random_B = random.randint(0,255), random.randint(0,255), random.randint(0,255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width), (random_R, random_G, random_B), -1)

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



            s_width, s_height = int((400-190)/2), int((800-269)/2)
            background[s_width:190 + s_width, s_height:269 + s_height, :] = Plate
            background = image_augmentation(background)

            if save:
                cv2.imwrite(self.save_path + label + ".jpg", background)
            else:
                cv2.imshow(label, background)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--img_dir", help="save image directory",
                    type=str, default="/home/truongdongdo/Desktop/CRNN-Keras/DB/per/test_1/")
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
A.Type_2(num_img, save=Save)
print("Type 2 finish")
A.Type_3(num_img, save=Save)
print("Type 3 finish")
A.Type_4(num_img, save=Save)
print("Type 4 finish")
A.Type_5(num_img, save=Save)
print("Type 5 finish")
# A.Type_6(num_img, save=Save)
# print("Type 6 finish")
