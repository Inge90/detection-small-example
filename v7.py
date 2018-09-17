#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 21:52:52 2018

@author: ingegrunberg
"""

import cv2
import numpy as np

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


#you can add your own video here.
#if you want to watch your laptop from the camerathen just add the number 0
#I got a video from the site : https://videos.pexels.com/videos/video-of-people-walking-855564
cap = cv2.VideoCapture('video2.mov')


version = cv2.__version__.split('.')[0]
print(version) 

while (True):
    ret, frame = cap.read((640,480))
    #edit video as line
    lines = cv2.Canny(frame, 100, 200)
    
    #brings the video out only in blue
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    colorOne = np.array([100,50,50])
    colorTwo = np.array([130,255,255])
    mask = cv2.inRange(hsv, colorOne, colorTwo)
    res = cv2.bitwise_and(frame,frame, mask=mask)
    
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(color, 1.1, 4)

    #finds an eye-catching face :D and and adds a frame
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),1)
        box_color = color[y:y+h, x:x+w]
        box_color = frame[y:y+h, x:x+w]
        
        #finds a smile on the face of emotion and adds a white frame
        smiles = smile_cascade.detectMultiScale(box_color)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(box_color,(sx,sy),(sx+sw,sy+sh),(255,255,255),1)

    cv2.imshow('res',res)
    cv2.imshow('lines',lines)
    cv2.imshow('detections', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()