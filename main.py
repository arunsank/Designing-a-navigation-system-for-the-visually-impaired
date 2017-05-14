import numpy as np
import random
import cv2
from numpy import inf
import imutils
import sys
import math


def corner_return(image):        
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #blr =cv2.bilateralFilter(image,9,40,75)
    
    rs= cv2.goodFeaturesToTrack(image,20,0.1,20,None,None,2,useHarrisDetector=False,k=0.04)                                
    #r,c,d= rs.shape
    
    #rs=np.int0(rs)
    #cv2.circle(image,(229,400),20,250,-1)     
                                
    return rs


def door_return(image):

    
    gr=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
    locs= cv2.goodFeaturesToTrack(gr,                    # img
                                20,                      # maxCorners
                                0.1,                     # qualityLevel
                                20,                      # minDistance
                                None,                    # corners, 
                                None,                    # mask, 
                                2,                       # blockSize, 
                                useHarrisDetector=False, # useHarrisDetector, 
                                k=0.04                   # k
                                )
    for k in locs:
        x,y= k.ravel()
    blr = np.uint8(gr)
    edges= cv2.Canny(blr,40,65,apertureSize = 3)
    edges = cv2.dilate(edges,None, iterations=1)
    edges = cv2.erode(edges,None, iterations=1)
    cont = cv2.findContours(edges.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #_,c,hier = cv2.findContours(edges.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cont = cont[0] if imutils.is_cv2() else cont[1]
    max_val=0
    k=0
    doors=[]
    knobs=[]
    for a in cont:
       #a= cn[0]
       #hie= cn[1]
       #perm= k
       #ch = hierarchyDataOfAContour[a]
       counter=0
       #epsilon = 0.331*cv2.arcLength(a,True)
       #a = cv2.approxPolyDP(a,epsilon,True)
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
            #image_cp1=image.copy()
            for i in range (len(locs)):
                pnts= locs[i]
                x2 = pnts[0,0]
                y2= pnts[0,1]
                
                distanc = cv2.pointPolygonTest(a,(x2,y2),True)
                
                
                if distanc>=0:
                    E_dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
                    
                    if (h/3)<=E_dist<h:
                        #cv2.circle(ima,(x2,y2),20,127,thickness=1, lineType=8, shift=0)
                        #cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                        doors.append((x,y,w,h))
                        knobs.append((x2,y2))
                    #else:
                        #cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                        
                else:
                    #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                    doors.append((x,y,w,h))
                    knobs.append(None)
                    #cv2.circle(image,(x2,y2),10,127,thickness=1, lineType=8, shift=0)
    return doors,knobs

'''
def ret_contours(image,locs):
    image_cp = image.copy()
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blr =cv2.bilateralFilter(image,9,75,75)
    edges= cv2.Canny(blr,40,65,apertureSize = 3)
    edges = cv2.dilate(edges,None, iterations=1)
    edges = cv2.erode(edges,None, iterations=1)
    cont = cv2.findContours(edges.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cont = cont[0] if imutils.is_cv2() else cont[1]
    #(cont,_) = contours.sort_contours(cont)
    max_val=0
    k=0
    for a in cont:
        m=0
        x,y,w,h = cv2.boundingRect(a)
    return (x,y,w,h)
'''

def calc_desc(img,desc):
    x,y,w,h  = desc
    buffer = 0
    kp = fast.detect(img[y-buffer:y+h+buffer,x-buffer:x+w+buffer], None)
    return len(kp)


def mergeRect(rect1,rect2):
    bx,by,bw,bh = rect2
    x,y,w,h = rect1
    bx1=bx+bw
    by1=by+bh
    x1=x+w
    y1=y+h
    nx=0;ny=0;nx1=0;ny1=0;
    nx = x if x<bx else bx
    ny = y if y<by else by
    nx1 = x1 if x1>bx1  else bx1
    ny1 = y1 if y1>by1  else by1
    return (nx,ny,(nx1-nx),(ny1-ny))

def nonMaxSup(lBoxes,tresh):
    w = frame.shape[1]
    h = frame.shape[0]
    tmpImg = np.array(np.random.rand(h,w),np.int32)
    tmpImg.fill(0)
    
    for i in lBoxes.keys():
        x,y,w,h = lBoxes[i];
        tmpImg[y:y+h,x:x+w] = i
    fringe = lBoxes.keys()
    while len(fringe)>0:
        i = fringe.pop();
        if i in lBoxes.keys():
            x,y,w,h = lBoxes[i]
            x1=x+w
            y1=y+h
            cx = (x+x1)/2
            cy = (y+y1)/2
            noConf = True
            j=0;
            if (noConf and y1+tresh<h and set(tmpImg[y1:y1+tresh,cx].flatten())!=set([0])):
                noConf = False
                confZone=set(tmpImg[y1:y1+tresh,cx].flatten())
            if (noConf and y-tresh>0 and set(tmpImg[y-tresh:y,cx].flatten())!=set([0])):
                noConf = False
                confZone=set(tmpImg[y-tresh:y,cx].flatten())
            if (noConf and x-tresh>0 and set(tmpImg[cy,x-tresh:x].flatten())!=set([0])):
                noConf = False
                confZone=set(tmpImg[cy,x-tresh:x].flatten())
            if (noConf and x1+tresh<w and set(tmpImg[cy,x1:x1+tresh].flatten())!=set([0])):
                noConf = False
                confZone=set(tmpImg[cy,x1:x1+tresh].flatten())
            if (noConf and y1+tresh<h and x1+tresh<w and set(tmpImg[y1:y1+tresh,x1:x1+tresh].flatten())!=set([0])):
                noConf = False
                confZone=set(tmpImg[y1:y1+tresh,x1:x1+tresh].flatten())
            if (noConf and y-tresh>0 and x-tresh>0 and set(tmpImg[y-tresh:y,x-tresh:x].flatten())!=set([0])):
                noConf = False
                confZone=set(tmpImg[y-tresh:y,x-tresh:x].flatten())
            if (noConf and y1+tresh<h and x-tresh>0 and set(tmpImg[y1:y1+tresh,x-tresh:x].flatten())!=set([0])):
                noConf = False
                confZone=set(tmpImg[y1:y1+tresh,x-tresh:x].flatten())
            if (noConf and y-tresh>0 and x1+tresh<w and set(tmpImg[y-tresh:y,x1:x1+tresh].flatten())!=set([0])):
                noConf = False
                confZone=set(tmpImg[y-tresh:y,x1:x1+tresh].flatten())
            if not noConf:
                confZone.discard(0)
                confZone.discard(i)
                for j in confZone:
                    lBoxes[i] = mergeRect(lBoxes[i],lBoxes[j])
                    (nx,ny,nw,nh) = lBoxes[i]
                    nx1 = nx+nw
                    ny1 = ny+nh
                    tmpImg[ny:ny1,nx:nx1] = i
                    #print x,y,x1,y1,bx,by,bx1,by1,nx,nx1,ny,ny1
                    #print i,j,lBoxes.keys()
                    lBoxes.pop(j)
                    contList[i].extend(contList[j])
                    if j in fringe:
                        fringe.remove(j)
                fringe.append(i)
                #print "dsadas",i,j,lBoxes.keys()
    return lBoxes


def isValidDoor(box):
    x,y,w,h = box
    if ((h>150) and (w>100 and w<200) and (h/float(w))>1 and (h/float(w))<3):
        if calc_desc(frame,(x,y,w,h))>50:
            return True;
    return False;

def isValidBox(box):
    x,y,w,h = box
    if ((h>30 and h<150) or (w>30 and w<150)):
        if calc_desc(frame,(x,y,w,h))>200:
            return True;
    return False;

def find_contor(label,k):
    kernel = np.ones((10,10),np.uint8)
    label_tmp = label == k
    label_tmp = label_tmp.astype(np.uint8)
    label_tmp = label_tmp.reshape(frameRight.shape)
    ret, thresh = cv2.threshold(label_tmp, 127, 255, 0)
    res2 = cv2.morphologyEx(label_tmp, cv2.MORPH_CLOSE, kernel)
    res2 = cv2.morphologyEx(label_tmp, cv2.MORPH_OPEN, kernel)
    im2,contours, hierarchy = cv2.findContours(label_tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def getposition(box):
    pos=""
    x,y,w,h  = box
    xc=x+w/2
    yc=y+h/2
    if yc-240<-80:
        pos+="T"
    if yc-240>80:
        pos+="B"
    if xc-320<-107:
        pos+="L"
    if xc-320>107:
        pos+="R"
    if pos=="":
        pos="C"
    return pos


def getValidPoints(point,thresh):
    x,y = point
    xn = x-thresh if x-thresh>0 else 0
    yn = y-thresh if y-thresh>0 else 0
    xn1 = x+thresh if x+thresh>0 else frame.shape[0]
    yn1 = y+thresh if y+thresh>0 else frame.shape[1]
    return ((xn,yn),(xn1,yn1))


def isDoorOpen(cnts,box,disp):
    points = []
    x,y,w,h  = box
    numOfPoints = 30
    energy=0
    blank=0
    nonBlank=0;
    waitCounter=0
    while(len(points)<numOfPoints):
        xp = random.randint(x,x+w)
        yp = random.randint(y,y+h)

        #print(xp,yp)
        #print(cv2.pointPolygonTest(cnt,(xp,yp),False))
        #print len(cnts)
        #print [cv2.pointPolygonTest(cnt,(xp,yp),False)>=0 for cnt in cnts]
        isValidPoint = any([cv2.pointPolygonTest(cnt,(xp,yp),False)>=0 for cnt in cnts])
        if isValidPoint and (yp,xp) not in points :
            points.append((yp,xp))
            energy+=disp[yp,xp]
            waitCounter=0
        waitCounter+=1

    totE =  energy/numOfPoints
    print totE
    return totE>100


def getDistancewithContour(cnts,box,disp):
    points = []
    x,y,w,h  = box
    dist = []
    infCount=0
    numOfPoints = 5
    while(len(points)<numOfPoints):
        xp = random.randint(x,x+w)
        yp = random.randint(y,y+h)
        #print(xp,yp)
        #print(cv2.pointPolygonTest(cnt,(xp,yp),False))
        #print len(cnts)
        #print [cv2.pointPolygonTest(cnt,(xp,yp),False)>=0 for cnt in cnts]
        isValidPoint = any([cv2.pointPolygonTest(cnt,(xp,yp),False)>=0 for cnt in cnts])
        if isValidPoint and (yp,xp) not in points :
            points.append((yp,xp))
            if disp[yp,xp]==0:
                infCount+=1
            else:
                dist.append(bf/disp[yp,xp])
    final = [x for x in dist if abs(x-np.mean(dist)) <= 1 * np.std(dist)]
    if infCount < numOfPoints-numOfPoints/2 and len(final)>0:
        retDist = sum(final)/len(final)
        return retDist
    else:
        return inf

    
isObsChk = False
isDoorChk = False
print len(sys.argv)
if len(sys.argv)==1:
    print "Not enough arguments"
    print "Usage python main.py <door/obs>"
    sys.exit(1)
if sys.argv[1]=="obs":
    isObsChk = True

if sys.argv[1]=="door":
    isDoorChk = True

leftInt = np.array([[1056.22474281452,2.12180012416206,464.811727631556],[0,1079.15545953137,167.86460716483],[0,0,1]])
rightInt = np.array([[1041.65499997853,-26.7186301189883,274.136149420335],[0,1010.94417804556,115.2322707182],[0,0,1]])
rightExt = np.array([-0.206102947808262, -0.552547032399206, 0.0012501669630201, 0.0324427475803376, 3.04532230656703])
leftExt = np.array([-0.673805020708009,5.14069106623387,0.0393615423664669,0.0209766146341484,-20.3847474068451])

f = (leftInt[0][0] + leftInt[1][1] +  rightInt[0][0] + rightInt[1][1])/40.0
b = 30

bf = f*b
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 3, (640,480))
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS,3);
cap1 = cv2.VideoCapture(2)
cap1.set(cv2.CAP_PROP_FPS,3);



ret,frameRight = cap.read()
a = np.array([range(0,frameRight.shape[1]),range(0,frameRight.shape[1])])
col = np.repeat(a,frameRight.shape[0]/2,axis=0)
b = np.array([range(0,frameRight.shape[0]),range(0,frameRight.shape[0])])
row = np.repeat(b,frameRight.shape[1]/2,axis=0)
row = row.transpose()
row = row.reshape((-1,1))
col = col.reshape((-1,1))
row = np.float32(row)
col = np.float32(col)
fast = cv2.FastFeatureDetector_create()
orb = cv2.ORB_create()

while(True):
    ret,frameRight = cap.read()
    frame = frameRight.copy()
    frameRight = cv2.medianBlur(frameRight,5)
    frameRight = cv2.cvtColor(frameRight,cv2.COLOR_BGR2GRAY)
    frameRight = cv2.equalizeHist(frameRight)
    #frameRight = cv2.undistort(frameRight, rightInt, rightExt, None,None)
    ret,frameLeft = cap1.read()
    frameLeft = cv2.medianBlur(frameLeft,5)
    frameLeft=cv2.cvtColor(frameLeft, cv2.COLOR_BGR2GRAY)
    frameLeft = cv2.equalizeHist(frameLeft)
    #frameLeft = cv2.undistort(frameLeft, leftInt, leftExt, None, None)
    stereo = cv2.StereoSGBM_create(0, 64, 10, 600, 2400, 20, 16, 1,  100, 20,True)
    disparity = stereo.compute(frameRight,frameLeft,cv2.CV_32F)
    np.bitwise_not(disparity,disparity)
    disparity = cv2.convertScaleAbs(disparity)

    h,  w = frameRight.shape[:2]
    #newcameramtxRight, roi=cv2.getOptimalNewCameraMatrix(frameRight,rightExt,(w,h),1)
    

    h,  w = frameLeft.shape[:2]
        #newcameramtxLeft, roi=cv2.getOptimalNewCameraMatrix(frame1,leftExt,(w,h),1,(w,h))



    #stereo = cv2.StereoSGBM(0, 96, 5, 600, 2400, 20, 16, 1,  100, 20,True)
    
    #stereo = cv2.StereoBM_create(1, 16, 15)
    #norm_coeff = 255/ disparity.max()
    #disparity= disparity*norm_coeff / 255
    #disparity = cv2.filterSpeckles(disparityMat,0,10,10)
        
    
    #kp = fast.detect(frameRight, None)
    #kp = orb.detect(frame,None)
    #kp, des = orb.compute(frame, kp)
    Z = frame.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)

    #Kvalues = np.column_stack((row,col,Z))
    Kvalues = Z
    Kvalues = np.array(Kvalues)
    #keypoints = detector.detect(frameRight)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5
    ret,label,center=cv2.kmeans(Kvalues,K,None,criteria,3,cv2.KMEANS_PP_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    #center = center[:,[2,3,4]]
    contors = []
    doors = []
    doorContor = []
    for i in range(0,K):
        cont = find_contor(label,i)
        contors.append(cont)
    res = center[label.flatten()]
    res2 = res.reshape((frame.shape))
    test_cont = find_contor(label,2)

    boxes = {}
    distList = {}
    index =0;
    doors=[]
    cornList = []
    contList = {}
    for i in range(0,K):
        tmp_cont = np.asarray(contors[i])
        for j in range(0,tmp_cont.shape[0]):
            cnt = tmp_cont[j]
            #cv2.drawContours(frame, [cnt], -1, (0,255,0), 1)
            x,y,w,h = cv2.boundingRect(cnt)
            box_temp = (x,y,w,h)
            if isValidBox(box_temp):
                dist = getDistancewithContour(cnt,box_temp,disparity)
                if dist != inf:
                    distList[index] = dist
                    boxes[index] = box_temp
                    contList[index] = [cnt]
                    index+=1;
            
            if isValidDoor(box_temp):
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                cC=0
                point = [[(box[1][0]+box[2][0])/2,(box[1][1]+box[2][1])/2]]
                box = np.concatenate((box, point), axis=0)
                point = [[(box[0][0]+box[3][0])/2,(box[0][1]+box[3][1])/2]]
                box = np.concatenate((box, point), axis=0)
                for index in range(len(box)):
                    point = getValidPoints(box[index],30)
                    #cv2.rectangle(frame,tuple(map(int,point[0])),tuple(map(int,point[1])),(0,255,0),2)
                    haris = corner_return(frame[point[0][1]:point[1][1],point[0][0]:point[1][0]])
                    if haris != None and len(haris)>0:
                        cC+=1    
                #box = np.int0(box)
                #cv2.drawContours(frame,[box],0,(0,0,255),2)
                if cC>=4:
                    doors.append(box_temp);
                    doorContor.append(cnt)
                #haris = corner_return(frame[y:y+h,x:x+w])
                #for rs in haris:
                #    x1,y1= rs.ravel()
                #    cornList.append((x1+x,y1+y))
                         
    #for i in boxes.keys():
    #    if not isValidBox(boxes[i]):
    #        boxes.pop(i)
    if isObsChk:
        boxes = nonMaxSup(boxes,5)
        for i in boxes.keys():
            x,y,w,h = boxes[i]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(disparity,(x,y),(x+w,y+h),(0,255,0),2)
            dist = getDistancewithContour(contList[i],boxes[i],disparity)
            if(dist!=inf):
                cv2.putText(frame,str(int(dist)),(x+w/2,y+h/2),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255))
        #else:
        #    if calc_desc(frame,(x,y,w,h))>10:
        #        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #epsilon = 0.1*cv2.arcLength(cnt,True)
        #approx = cv2.approxPolyDP(cnt,epsilon,True)
        #cv2.drawContours(frame, [cnt], -1, (0,255,0), 1)
    #cv2.drawContours(frame, test_cont, -1, (0,255,0), 1)
    #doors,knobs = door_return(frame)
    if isDoorChk:
        for index in range(len(doors)):
            x,y,w,h = doors[index]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            status = isDoorOpen(doorContor[index],doors[index],disparity)
            if status:
                cv2.putText(frame,"Close",(x,y+h/2),cv2.FONT_HERSHEY_TRIPLEX,2,(0,0,255))
            else:
                cv2.putText(frame,"Open",(x,y+h/2),cv2.FONT_HERSHEY_TRIPLEX,2,(0,0,255))
    #for rs in cornList:
    #    rs = map(int,rs)
    #    x,y = rs
    #    cv2.circle(frame,(x,y),5,127,-1)
    #frame = cv2.drawKeypoints(frame, kp, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('frame',frame)
    #print(set(disparity.flatten()))
    #cv2.imshow('disparity',disparity)
    '''
    # find and draw the keypoints
    kp = fast.detect(gray, None)
    keypoints = detector.detect(gray)
    print(keypoints)
    im = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('frame',im)
    '''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
