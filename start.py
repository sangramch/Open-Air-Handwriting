import cv2
from collections import deque
import imutils
import numpy as np
import math

def eucdist(a,b):
    x1=a[0]
    y1=a[1]
    x2=b[0]
    y2=b[1]
    dist=math.sqrt( ((x2-x1)**2) + ((y2-y1)**2) )
    return dist

cap=cv2.VideoCapture(0)
pts=deque(maxlen=256)
counter=0

sensitivity=15
lower_color=(100,75, 50)
upper_color=(110, 255, 255)


j=0

while True:
    _,frame=cap.read()
    frame=cv2.flip(frame,1)
    
    if frame is None:
        break

    blurred=cv2.GaussianBlur(frame,(11,11),0)
    hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)

    mask=cv2.inRange(hsv,lower_color,upper_color)
    mask=cv2.erode(mask, None, iterations=2)
    mask=cv2.dilate(mask, None, iterations=2)
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    
    if len(cnts)>0:
        c = max(cnts, key=cv2.contourArea)
        ((x,y),radius)=cv2.minEnclosingCircle(c)
        M=cv2.moments(c)
        center=((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        
        cv2.circle(frame,center, 5, (0,0,255), -1)
        pts.appendleft(center)
        
        for i in np.arange(1, len(pts)):
            if pts[i-1] is None or pts[i] is None:
                continue
            
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 2)

        if (counter>30):
            avg=eucdist(pts[0],pts[30])
            if avg<4:
                alpha=np.zeros((480,640,3))
                for i in np.arange(1, len(pts)):
                    alpha=cv2.line(alpha, pts[i - 1], pts[i], (0, 0, 255), 2)
                
                filename="SaveFile"+str(j)+".png"
                cv2.imwrite((filename),alpha)
                j+=1
                pts.clear()
                counter=0
                center=None
                print("Saved: "+filename)
        counter+=1
            
            
    
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break



cap.release()
cv2.destroyAllWindows()