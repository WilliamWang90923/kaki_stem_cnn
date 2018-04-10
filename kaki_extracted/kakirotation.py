#coding=utf-8
import pydot
import numpy as np  
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import argparse
import cv2
import math
import os
import sys
 																								
img_path = '/home/cherry/kaki_extracted/test/'

#list = os.listdir(img_path)
#image = cv2.imread(img_path + "testnew003.png", cv2.IMREAD_UNCHANGED)

#points = xmlreader_test.getXML('testnew003.xml')
#(h, w) = image.shape[:2]

def rotate(image, angle, points,center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center,angle,scale)
    #print(M)
    rotated = cv2.warpAffine(image,M,(w,h))
    point2 = [[0,0],[0,0],[0,0],[0,0],[0,0]]
    for point in points: 
        dx = point[0]-center[0]
        dy = point[1]-center[1]
        point2[points.index(point)] = [0,0]
        point2[points.index(point)][0] = int(dx*math.cos(angle*2*3.1415926/360) + dy*math.sin(angle*2*3.1415926/360)+center[0]) 
        point2[points.index(point)][1] = int(-dx*math.sin(angle*2*3.1415926/360) + dy*math.cos(angle*2*3.1415926/360)+center[1]) 
    return rotated,point2



#ap=argparse.ArgumentParser()
#ap.add_argument("-i","--image",required=True,help="Path to the image")
#args=vars(ap.parse_args())

if __name__ == "__main__":
    rotated,point2=rotate(image,60,points)
    cv2.circle(image,(points[1][0],points[1][1]),2,(55,255,155),2)
    cv2.circle(image,(points[3][0],points[3][1]),2,(55,255,155),2)
    cv2.circle(image,(points[4][0],points[4][1]),2,(55,255,155),2)
    cv2.circle(image,(points[2][0],points[2][1]),2,(55,255,155),2)
    cv2.circle(image,(points[0][0],points[0][1]),2,(55,255,155),2)
    cv2.imshow("original",image)
    cv2.circle(rotated,(point2[1][0],point2[1][1]),2,(55,255,155),2)
    cv2.circle(rotated,(point2[3][0],point2[3][1]),2,(55,255,155),2)
    cv2.circle(rotated,(point2[4][0],point2[4][1]),2,(55,255,155),2)
    cv2.circle(rotated,(point2[2][0],point2[2][1]),2,(55,255,155),2)
    cv2.circle(rotated,(point2[0][0],point2[0][1]),2,(55,255,155),2)
    cv2.imshow("rotate degrees",rotated)

    cv2.waitKey()
    cv2.destroyAllWindows()


