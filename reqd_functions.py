
import cv2
import numpy as np
import math

def eucdist(a,b):
    x1=a[0]
    y1=a[1]
    x2=b[0]
    y2=b[1]
    dist=math.sqrt( ((x2-x1)**2) + ((y2-y1)**2) )
    return dist


def findROI(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    indices=np.nonzero(gray)
    
    top_x=np.min(indices[0])
    bottom_x=np.max(indices[0])
    
    left_y=np.min(indices[1])
    right_y=np.max(indices[1])
    
    height=bottom_x-top_x
    width=right_y-left_y
    
    if height>=width:
        if top_x>10:
            top_x=top_x-10
            height+=10
    
    
        if bottom_x<630:
            bottom_x=bottom_x+10
            height+=10
    
    
        if left_y>=int((height-width)/2) and right_y<=640-int((height-width)/2):
            left_y-=int((height-width)/2)
            right_y+=int((height-width)/2)
    
        elif (left_y)<=int((height-width)/2):
            right_y+=((height-width))
    
        elif (right_y)>=640-int((height-width)/2):
            left_y-=((height-width))
    
    
        image=image[top_x:bottom_x,left_y:right_y]
    
    
    else:
        if left_y>10:
            left_y=left_y-10
            width+=10
    
    
        if right_y<630:
            right_y=right_y+10
            width+=10
    
    
        if top_x>=int((width-height)/2) and bottom_x<=640-int((width-height)/2):
            top_x-=int((width-height)/2)
            bottom_x+=int((width-height)/2)
    
        elif (top_x)<=int((width-height)/2):
            bottom_x+=((width-height))
    
        elif (bottom_x)>=640-int((width-height)/2):
            top_x-=((width-height))
    
    
        image=image[top_x:bottom_x,left_y:right_y]

    return image
    
    
  