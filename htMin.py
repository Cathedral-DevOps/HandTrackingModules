import cv2

import mediapipe as mp
import time

cap = cv2.VideoCapture(1)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to rgb for process
    results = hands.process(imgRGB) #process
    print(results.multi_hand_landmarks) #print detected
    
    #nested if loop for drawing points on hands
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms)

    cv2.imshow("Image", img)
    cv2.waitKey(1)