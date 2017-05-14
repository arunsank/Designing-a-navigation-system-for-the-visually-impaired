# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 08:27:05 2017

@author: Arun Ram
"""
import os
import numpy as np
import cv2
cv2.ocl.setUseOpenCL(False)
import scipy
from scipy import spatial
from matplotlib import pyplot as plt
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse
import imutils
import math


os.chdir('C:\\Users\\Arun Ram\\Desktop\\Vision project')

        
def corner_return(image):
    gr=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    blr = cv2.GaussianBlur(image,(5,5),0)
    
    locs= cv2.goodFeaturesToTrack(gr,20,0.1,20,None,None,2,useHarrisDetector=False,k=0.04)
    
    for k in locs:
        x,y= k.ravel()
        image1 = image.copy()
        cv2.circle(image1,(x,y),5,127,-1)
    blr = np.uint8(blr)
    edges= cv2.Canny(blr,40,65,apertureSize = 3)
    edges = cv2.dilate(edges,None, iterations=1)
    edges = cv2.erode(edges,None, iterations=1)
    cont = cv2.findContours(edges.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cont = cont[0] if imutils.is_cv2() else cont[1]
    max_val=0
    k=0
    for a in cont:
       #a= cn[0]
       #hie= cn[1]
       #perm= k
       #ch = hierarchyDataOfAContour[a]
       counter=0
       epsilon = 0.1*cv2.arcLength(a,True)
       a = cv2.approxPolyDP(a,epsilon,True)
       k=cv2.contourArea(a)
       x,y,w,h = cv2.boundingRect(a)
       #area= cv2.contourArea(a)
       #Mid = cv2.moments(a)
       #mx= int(Mid['m10']/Mid['m00'])
       #my = int(M['m01']/M['m00'])
       #counter =0
       for i in range(len(locs)):
           z= locs[i]
           x1 = z[0,0]
           y1 = z[0,1]
           rat = (h/w)
           
           
           if (x<=x1<=(x+w)) and (y<=y1<=(y+h)) and ((w/h)<0.8) and (2<rat<5) and (h>(2*w)):
               counter+=1
             
       if (counter>max_val):
            max_val=counter
            for i in range (len(locs)):
                pnts= locs[i]
                x2 = pnts[0,0]
                y2= pnts[0,1]
                
                distanc = cv2.pointPolygonTest(a,(x2,y2),True)
                
                if distanc>=0:
                    E_dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
                    
                    if (h/3)<=E_dist<h:
                        cv2.circle(image,(x2,y2),20,127,thickness=1, lineType=8, shift=0)
                        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
                    #else:
                        #cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                #uncomment below other rectangles        
                #else:
                    #cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
                
                    #cv2.circle(image,(x2,y2),10,127,thickness=1, lineType=8, shift=0)
                    
    return image1,locs


#Main
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
cap = cv2.VideoCapture(0)
#cap1 = cv2.VideoCapture()
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #ret, frame1 = cap1.read()
    #result= test(frame)
    result, corn_loc= corner_return(frame)
    #result1, corn_loc= corner_return(frame1)
    #final_results =ret_contours(frame,corn_loc)


    out.write(result)
    # Display the resulting frame
    cv2.imshow('frame',result)
    #cv2.imshow('frame1',result1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
# When everything done, release the capture
out.release()        
cap.release()
cv2.destroyAllWindows()



