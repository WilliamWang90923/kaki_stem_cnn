#coding=utf-8
import pydot
import numpy as np  
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import argparse
import cv2
import math
import os 																								
img_path = '/home/cherry/keras/data/train/image3/'

list = os.listdir(img_path)
image = cv2.pyrDown(cv2.imread(img_path + "001.png", cv2.IMREAD_UNCHANGED))

point = [400,200]
(h, w) = image.shape[:2]

def rotate(image, angle, point,center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center,angle,scale)
    print(M)
    rotated = cv2.warpAffine(image,M,(w,h))
    dx = point[0]-center[0]
    dy = point[1]-center[1]
    point2 = [0,0]
    point2[0] = dx*math.cos(angle*2*3.1415926/360) + dy*math.sin(angle*2*3.1415926/360)+center[0] 
    point2[1] = -dx*math.sin(angle*2*3.1415926/360) + dy*math.cos(angle*2*3.1415926/360)+center[1] 
    return rotated,point2



#ap=argparse.ArgumentParser()
#ap.add_argument("-i","--image",required=True,help="Path to the image")
#args=vars(ap.parse_args())

rotated,point2=rotate(image,-160,point)
cv2.circle(image,(point[0],point[1]),10,(55,255,155),2)
cv2.imshow("original",image)
cv2.circle(rotated,(int(point2[0]),int(point2[1])),10,(55,255,155),2)
cv2.imshow("rotate degrees",rotated)

cv2.waitKey()
cv2.destroyAllWindows()


