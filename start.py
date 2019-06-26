import cv2
from collections import deque
import imutils
import numpy as np
import pytesseract

from reqd_functions import findROI,eucdist



cap=cv2.VideoCapture(0)
pts=deque(maxlen=256)
counter=0
nocnt_frames=0
bufclean=True

sensitivity=15
lower_color=(100,75, 50)
upper_color=(110, 255, 255)

alphalist=['A']

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
        nocnt_frames=0
        bufclean=False
        c = max(cnts, key=cv2.contourArea)
        ((x,y),radius)=cv2.minEnclosingCircle(c)
        M=cv2.moments(c)
        center=((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        
        pts.appendleft(center)
        
        for i in np.arange(1, len(pts)):
            if pts[i-1] is None or pts[i] is None:
                continue
            
            cv2.line(frame, pts[i - 1], pts[i], (255, 255, 255), 15)

        if (counter>50):
            avg=eucdist(pts[0],pts[50])
            if avg<4:
                alpha=np.zeros((640,640,3),dtype='uint8')
                for i in np.arange(1, len(pts)):
                    cv2.line(alpha, pts[i - 1], pts[i], (255, 255, 255), 15)
                
                filename="SaveFile"+str(j)+".png"
                roi=findROI(alpha)
                roi=cv2.bitwise_not(roi)
                
                text=pytesseract.image_to_string(roi,config='-l eng --oem 0 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ<')
                if text=='<':
                    alphalist.pop()
                    print("Deleted")
                else:
                    alphalist.append(text)
                
                j+=1
                pts.clear()
                counter=0
                center=None
                print("Detected")
                
        cv2.circle(frame,center, 5, (0,0,255),8)
        counter+=1
    
    if len(cnts)==0:
        if not bufclean:
            nocnt_frames+=1
            if nocnt_frames>40:
                pts.clear()
                counter=0
                center=None
                nocnt_frames=0
                bufclean=True
                print("Cleared buffer")

    dispim=np.hstack((frame, cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)))
    
    alphabet="".join(alphalist)
    cv2.putText(dispim, alphabet, (650, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255),3, lineType=cv2.LINE_AA)
    
    cv2.imshow('frame',dispim)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break



cap.release()
cv2.destroyAllWindows()