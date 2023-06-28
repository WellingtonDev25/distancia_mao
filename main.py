import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import cvzone

# Webcam
video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
video.set(3, 1280)
video.set(4, 720)

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Find Function
# x is the raw distance y is the value in cm
distPixels = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
distCM = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coef = np.polyfit(distPixels, distCM, 2)  # y = Ax^2 + Bx + C

# Loop
while True:
    check, img = video.read()
    hands = detector.findHands(img, draw=False)

    if hands:
        lmList = hands[0]['lmList']
        x, y, w, h = hands[0]['bbox']
        x1, y1,_ = lmList[5]
        x2, y2,_ = lmList[17]

        #ponto de referencia
        dist = int(abs(x2-x1))

        #extraindo os coeficientes
        A, B, C = coef
        distCM = (A * dist ** 2) + (B * dist) + C

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
        cvzone.putTextRect(img, f'{int(distCM)} cm', (x+5, y-10))

    cv2.imshow("Image", img)
    cv2.waitKey(1)