# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 15:15:30 2021

@author: PraveenAnandhanathan
"""

import cv2
import numpy as np
import time
import threading
import math
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from pygame import mixer
from email.message import EmailMessage
import smtplib
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

alert = False
report = 0

def play_alarm():
    mixer.init()
    mixer.music.load('dataset/alarm.mp3')
    mixer.music.play()
    time.sleep(10)
    mixer.music.stop()
    
def send_mail():
    receiver = 'prave.anand124@gmail.com'
    subject = '...!! FIRE ALERT !!...'
    body = 'Warning!! Forest Fire has been detected. Take control ASAP!'
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login('tamilcipher@gmail.com', 'Abcdefgh1.')
    email = EmailMessage()
    email['From'] = 'Forest Fire Detector'
    email['To'] = receiver
    email['Subject'] = subject
    email.set_content(body)
    server.send_message(email)
    print('Imtimatated through mail to ' + receiver)
    server.close()

video = cv2.VideoCapture("dataset/ff2.mp4")

while True:
    check, frame = video.read()
    
    #MOBILE NET
    frame = cv2.resize(frame, (224, 224))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_array = img_to_array(frame_rgb)
    preprocess = tf.keras.applications.mobilenet.preprocess_input(frame_array)
    model = load_model("forestfire_model.h5")
    output = np.array(preprocess)
    output = output.reshape(1,224,224,3)
    output = model.predict(output)
    # print(output)
    if output[0][0] > output[0][1]:
        percent = str(math.floor(output[0][0]*100))
        print ("Amount of fire: " + percent +"%" )
        report = report + 1
    if report >= 1:
     	if alert == False:
             threading.Thread(target=play_alarm).start()
             threading.Thread(target=send_mail).start()
             alert = True
        
    
    #HSV MODEL
    # frame = cv2.resize(frame, dsize=(500,500))
    # gaussianblur = cv2.GaussianBlur(frame, (15,15),0) #GAUSSIAN BLUR
    # hsvframe = cv2.cvtColor(gaussianblur, cv2.COLOR_BGR2HSV) #HSV TO DECTECT FIRE EASILY
    # lower = np.array([18, 50, 50], dtype="uint8")
    # upper = np.array([35, 255, 255], dtype="uint8") #Color Range for Fire
    # mask = cv2.inRange(hsvframe, lower, upper)
    # frame = cv2.bitwise_and(frame, hsvframe, mask=mask)
    # amount = cv2.countNonZero(mask)
    
    # if int(amount) > 5000:
    #     print("alert: Fire")
    #     report = report + 1
    # if report >= 1:
    #  	if alert == False:
    #         threading.Thread(target=play_alarm).start()
    #         threading.Thread(target=send_mail).start()
    #         alert = True
    
    label = "FIRE" if output[0][0] > output[0][1] else "No FIRE"
    color = (0, 0, 255) if label == "FIRE" else (0, 255, 0)
    label = "{}: {:.2f}%".format(label, max(output[0][0], output[0][1]) * 100)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    frame = cv2.resize(frame, (500, 500))
    
    cv2.imshow("Forest Fire Detector",frame)
    if check == False:
        break
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()
