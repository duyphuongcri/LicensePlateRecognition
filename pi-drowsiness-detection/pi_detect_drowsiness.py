#!/usr/bin/env python3
# coding: utf-8
# USAGE
# python pi_detect_drowsiness.py --cascade haarcascade_frontalface_default.xml --shape-predictor shape_predictor_68_face_landmarks.dat
# python pi_detect_drowsiness.py --cascade haarcascade_frontalface_default.xml --shape-predictor shape_predictor_68_face_landmarks.dat --alarm 1

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from threading import Thread
from pygame import mixer
import serial
from time import strftime
from datetime import datetime
import math

path='/home/truongdongdo/Desktop/Drowsy_pi_Final/pi-drowsiness-detection/Hinh_Tai_Xe/'

def sound_welcome():
    mixer.init()
    mixer.music.load('sound_welcome.mp3')
    mixer.music.play()

def sound_canh_bao_ngu_gat():
    mixer.init()
    mixer.music.load('sound_ngu_gat.mp3')
    mixer.music.play()

def sound_canh_bao_nhin_thang():
    mixer.init()
    mixer.music.load('sound_nhin_thang.mp3')
    mixer.music.play()

def sound_3h_lien_tuc():
    mixer.init()
    mixer.music.load('sound_3h_lien_tuc.mp3')
    mixer.music.play()

def sound_mat_phanh():
    mixer.init()
    mixer.music.load('sound_mat_phanh.mp3')
    mixer.music.play()


def euclidean_dist(ptA, ptB):
	# compute and return the euclidean distance between the two
	# points
	return np.linalg.norm(ptA - ptB)

def set_time(h,m,s):
    set_time=h*3600+m*60+s
    return set_time

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = euclidean_dist(eye[1], eye[5])
	B = euclidean_dist(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = euclidean_dist(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

def Nhan_mat_set_time(COUNTER_FACE_THRES):
	COUNTER_FACE=0
	while COUNTER_FACE < COUNTER_FACE_THRES:
		frame = vs.read()
		frame = imutils.resize(frame, width=320)    #450, cang nho xu ly cang nhanh
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))
		gray = clahe.apply(gray)
		# Nhan dien va trich xuat khuon mat
		rects = detector(gray, 0)
		# rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
		# flags=cv2.CASCADE_SCALE_IMAGE)
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
 
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
			# do a bit of cleanup
			cv2.destroyAllWindows()
			vs.stop()
		if len(rects) > 0:
			COUNTER_FACE += 1
		else:
			COUNTER_FACE=0
		print(COUNTER_FACE)
	else:
		# t1 = Thread(sound_welcome())
		# t1.deamon = True
		# t1.start()
		sound_welcome()
		print("[WELCOME] Driver start to drive the car at")
		start_time = datetime.now()
		print(start_time)
		name =path+ str(start_time)+ '.jpg'
		cv2.imwrite(name, frame)
	return start_time
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help = "path to where the face cascade resides")
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=int, default=0,
	help="alarm: connect with arduino via Serial")
args = vars(ap.parse_args())

# check to see if we are using GPIO/TrafficHat as an alarm
if args["alarm"] > 0:

	# Khai bao Serial
	# Tao file duong dan am thanh
	#serial transports
	ser = serial.Serial('/dev/ttyACM0',115200)
	print("[INFO] using Arduino alarm...")

select_para=0
EYE_AR_THRESH=0
EYE_AR_CONSEC_FRAMES=0
SET_HR=0
SET_MIN=0
SET_SEC=0
COUNTER_FACE_THRES=0
COUNTER_NON_FACE_THRES_NHIN_THANG = 0
COUNTER_NON_FACE_THRES_RESET=0
time_repeat=0
so_lan_nhin_thang=0


select_para=int(input(" Ban co muon thay doi thong so khong ? -0.Khong -1.Co "))
if select_para==0:
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
	EYE_AR_THRESH = 0.22 		#Nguong EAR
	EYE_AR_CONSEC_FRAMES = 15	#So frame nham mat lien tuc
	# Set_time - Dat truoc thoi gian canh bao tai xe lai xe lien tuc
	SET_HR=0					#Gio
	SET_MIN=1					#Phut
	SET_SEC=0					#Giay

	COUNTER_FACE_THRES=15		#So khung hinh lien tuc co khuon mat de chuong trinh bat dau hoat dong
	COUNTER_NON_FACE_THRES_NHIN_THANG = 15 	#So khung hinh khong nhan thay mat de canh bao NHIN THANG
	COUNTER_NON_FACE_THRES_RESET=150		#So khung hinh nhan biet tai xe ra khoi xe
	time_repeat=45							#Thoi gian nhac lai canh bao tai xe lai xe lien tuc qua so gio(theo giay (s))
	so_lan_nhin_thang=3

else:
	EYE_AR_THRESH = float(input("Thiet lap nguong EAR (0.22): "))
	print(EYE_AR_THRESH)
	EYE_AR_CONSEC_FRAMES = int(input("Thiet lap so frame lien tuc nham mat --> Buon ngu (15): "))
	print(EYE_AR_CONSEC_FRAMES)
	print("Thiet lap thoi gian canh bao tai xe lai xe lien tuc (HH:MM:SS) (00:03:00): ")
	# Set_time
	SET_HR=int(input("Gio (00): "))
	SET_MIN=int(input("Phut (03): "))
	SET_SEC=int(input("Giay (00): "))
	print(" Da thiet lap: {}:{}:{}".format(SET_HR,SET_MIN,SET_SEC))

	COUNTER_FACE_THRES=int(input("So khung hinh lien tuc co khuon mat de chuong trinh bat dau hoat dong (15): "))
	print(COUNTER_FACE_THRES)
	COUNTER_NON_FACE_THRES_NHIN_THANG = int(input("So khung hinh khong nhan thay mat de canh bao NHIN THANG (15): "))
	print(COUNTER_NON_FACE_THRES_NHIN_THANG)
	so_lan_nhin_thang= int(input("So lan canh bao nhin thang truoc khi roi khoi xe (3): "))
	print(so_lan_nhin_thang)
	COUNTER_NON_FACE_THRES_RESET=int(input("So khung hinh nhan biet tai xe ra khoi xe (150): "))
	time_repeat=int(input("Thoi gian nhac lai canh bao tai xe lai xe lien tuc qua so gio(theo giay (s)) (45): "))
	print(time_repeat)
	print("-------------------------------------------------")
	print("Da thiet lap cac thong so!")
	time.sleep(2.0)

COUNTER = 0
COUNTER_FACE=0
COUNTER_NON_FACE=0		
canh_bao=0
canh_bao_3h=0
ALARM_ON = False
hist_size=(4,4)
hist_limit=1.0
src_cam=1					#Chon cong cho Camera (pi: 0, laptop:1)
duration=0

# lay thong tin detect mat va facial 68
# load OpenCV's Haar cascade for face detection (which is faster than
# dlib's built-in HOG detector, but less accurate), then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
# detector = cv2.CascadeClassifier(args["cascade"])
predictor = dlib.shape_predictor(args["shape_predictor"])

# lay thong tin vung mat
# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=src_cam).start()                            #Dung USB camera
# vs = VideoStream(usePiCamera=True).start()               #Dung Pi camera
time.sleep(1.0)

set_time = set_time(SET_HR,SET_MIN,SET_SEC)
print(set_time)

start_time = Nhan_mat_set_time(COUNTER_FACE_THRES)
  
# Vong lap while true
# loop over frames from the video stream
while True:
	state=ser.readline()
	print(state)
	if state == "phanh":
		sound_mat_phanh()
	# Lay frame anh, resize va chuyen thanh anh gray
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=320)    #450, cang nho xu ly cang nhanh
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
	# cv2.imshow("gray_org", gray)
	# create a CLAHE object (Arguments are optional).
	clahe = cv2.createCLAHE(clipLimit=hist_limit, tileGridSize=hist_size)
	gray = clahe.apply(gray)
	# cv2.imshow("gray", gray)

	# Nhan dien va trich xuat khuon mat
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	# rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
	# 	minNeighbors=5, minSize=(30, 30),
	# 	flags=cv2.CASCADE_SCALE_IMAGE)
	# print(len(rects))
	if len(rects) > 0 :
		canh_bao=0
		current_time= datetime.now()
		duration=(current_time - start_time).total_seconds()
		# print(duration)
		print("Thoi gian canh bao lai xe lien tuc: {}(s)".format(set_time))
		if duration >=set_time and canh_bao_3h==0:
			sound_3h_lien_tuc()
			canh_bao_3h=1
			if args["alarm"] > 0:
											
				#Send data
				send3h = "3h        " + str(datetime.now().strftime('%H:%M %d/%m/%Y'))
				ser.write(send3h.encode())
				time.sleep(0.1)

			cv2.putText(frame, "LAI XE 3H45", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
			print(datetime.now().strftime('%H:%M %d/%m/%Y'))
		if duration >=set_time and round(duration- set_time) % time_repeat==0:
			canh_bao_3h=0 
		COUNTER_NON_FACE=0
		for rect in rects:
		# for (x, y, w, h) in rects:
		# construct a dlib rectangle object from the Haar cascade
		# bounding box
			# rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
			# Phan tich facial 68 tu vung vua cat
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# Trich xuat diem cua mat trai va mat phai
			# extract the left and right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for both eyes
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)

			# average the eye aspect ratio together for both eyes
			ear = (leftEAR + rightEAR) / 2.0
					
			# Dem thoi gian
			# check to see if the eye aspect ratio is below the blink
			# threshold, and if so, increment the blink frame counter
			if ear < EYE_AR_THRESH:
				COUNTER += 1
				# Hien thi chu len man hinh
				# draw an alarm on the frame
				# Ve vung mat
				# compute the convex hull for the left and right eye, then
				# visualize each of the eyes
				leftEyeHull = cv2.convexHull(leftEye)
				rightEyeHull = cv2.convexHull(rightEye)
				cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
				cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

				# if the eyes were closed for a sufficient number of
				# frames, then sound the alarm
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# print("Phat hien tai xe buon ngu")
				# if the alarm is not on, turn it on
					if not ALARM_ON:
						ALARM_ON = True
								
						#Truyen du lieu sang arduino
						# Phat am thanh
													
						# check to see if an alarm file was supplied,
						# and if so, start a thread to have the alarm
						# sound played in the background
						if args["alarm"] > 0:
							t = Thread(sound_canh_bao_ngu_gat())
							t.deamon = True
							t.start()
												
							#Send data
							send = "warning " + str(datetime.now().strftime('%H:%M %d/%m/%Y'))
							ser.write(send.encode())
							time.sleep(0.1)

					cv2.putText(frame, "TAI XE BUON NGU", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
					print(datetime.now().strftime('%H:%M %d/%m/%Y'))

					# otherwise, the eye aspect ratio is not below the blink
					# threshold, so reset the counter and alarm
			else:
				COUNTER = 0
				ALARM_ON = False
				# Ve vung mat
				# compute the convex hull for the left and right eye, then
				# visualize each of the eyes
				leftEyeHull = cv2.convexHull(leftEye)
				rightEyeHull = cv2.convexHull(rightEye)
				cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
				cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

				# draw the computed eye aspect ratio on the frame to help
				# with debugging and setting the correct eye aspect ratio
				# thresholds and frame counters
			cv2.putText(frame, "EAR: {:.3f}".format(ear), (220, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			cv2.putText(frame, "{:.0f}".format(COUNTER), (280, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
	else:
		COUNTER_NON_FACE +=1

	if COUNTER_NON_FACE > COUNTER_NON_FACE_THRES_RESET and canh_bao >= so_lan_nhin_thang:
		COUNTER_NON_FACE=0
		canh_bao_3h=0
		canh_bao=0
		print("Tai xe da ra khoi xe")
		time.sleep(2.0)
		start_time = Nhan_mat_set_time(COUNTER_FACE_THRES)

	elif COUNTER_NON_FACE > COUNTER_NON_FACE_THRES_NHIN_THANG and canh_bao < so_lan_nhin_thang:
		sound_canh_bao_nhin_thang()
		canh_bao +=1
		COUNTER_NON_FACE=0
		time.sleep(4.0)

	# show the frame
	cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (10, 230),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
	h = math.floor(duration/3600)
	m = math.floor((duration-h*3600)/60)
	s = math.floor(duration - h*3600-m*60)
	string_time=name = str(h)+':'+str(m)+':'+str(s)
	date_time_obj = datetime.strptime(string_time, '%H:%M:%S')
	print('Time:', date_time_obj.time())  
	cv2.putText(frame," LAI XE LIEN TUC: %s" %date_time_obj.time(), (5, 200), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
		# do a bit of cleanup
		cv2.destroyAllWindows()
		vs.stop()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()