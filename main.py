import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

video = cv2.VideoCapture(0,cv2.CAP_DSHOW)

video.set(3,1280)
video.set(4,720)

detector = HandDetector(detectionCon=0.8,maxHands=1)

distPixels = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
distCM = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coef = np.polyfit(distPixels,distCM,2)

while True:
    check,img = video.read()
    hands = detector.findHands(img,draw=False)

    if hands:
        lmlist = hands[0]['lmList']
        x,y,w,h = hands[0]['bbox']
        x1,y1,_ = lmlist[5]
        x2,y2, _ = lmlist[17]
        dist = (abs(x2 - x1))
        A,B,C = coef
        distCMT = (A*dist**2)+(B*dist)+C

        print(dist,distCMT)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),3)
        cvzone.putTextRect(img,f'{int(distCMT)} cm',(x+5,y-10))

    cv2.imshow('Imagem',img)
    cv2.waitKey(1)
