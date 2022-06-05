#!/usr/bin/python
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

width, height = (
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
    )
print(f"Camera dimensions: {width, height}")
print(f"Camera FPS: {cap.get(cv2.CAP_PROP_FPS)}")

import time
start = time.monotonic()
count= 0
while 1:
    ret, image = cap.read()
    #....
    count+=1
    if count%30==0:
        end = time.monotonic()
        print(30/(end-start))
        start = end

cap.release()
cv2.destroyAllWindows()
