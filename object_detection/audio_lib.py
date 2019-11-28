import cv2
from functools import wraps
from pygame import mixer
import time
from threading import Thread

from playsound import playsound


cap = cv2.VideoCapture(0)

def sound_3h_lien_tuc():
    # mixer.init()
    # mixer.music.load('sound_3h_lien_tuc.mp3')
    # mixer.music.play() # play once
    playsound('sound_3h_lien_tuc.mp3')
    # time.sleep(5)
count = 0

while 1:
    ret, img = cap.read()
    count +=1
    cv2.imshow('img', img)
    if count == 10:
        # t = Thread(sound_3h_lien_tuc())
        # t.deamon = True
        # t.start()
        # time.sleep(10)

        sound_3h_lien_tuc()
        count = 0 

    
    k = cv2.waitKey(30) & 0xff


cap.release()
cv2.destroyAllWindows()
