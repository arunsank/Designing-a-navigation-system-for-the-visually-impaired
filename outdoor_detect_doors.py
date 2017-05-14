#Uses Viola-Jones classifier for door detection
# author: gnagaraj

import numpy as np
import cv2


cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture(0)
door_cascade=cv2.CascadeClassifier('cascade.xml')
while True:
    ret,frame=cap.read()
    doors=door_cascade.detectMultiScale(frame, 1.3, 5)
    for (x,y,w,h) in doors:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
