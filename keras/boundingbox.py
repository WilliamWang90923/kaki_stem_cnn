#coding=utf-8
import pydot
import numpy as np  
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import cv2
from xmlreader_test import getXML 																								
img_path = '/home/cherry/keras/data/train/image3/'
list = os.listdir(img_path)
f = open('/home/cherry/kaki_extracted/labelnew.txt','w')
for i in list:
    img = cv2.imread(img_path + i, cv2.IMREAD_UNCHANGED)
    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY),180,255,cv2.THRESH_BINARY_INV)
   # cv2.imwrite('/home/cherry/testnew'+i,thresh)
    contours, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(i)
    L = getXML(i.split('.')[0]+'.xml')
    for c in contours:    
        x, y, w, h = cv2.boundingRect(c) 
        #cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        if (h > 100 and w > 150):  
            cropImg = img[y:y+h,x:x+w]
            print(x,y)
            
            L[0] = L[0] - x
            L[2] = L[2] - x
            L[4] = L[4] - x
            L[6] = L[6] - x
            L[8] = L[8] - x
            L[1] = L[1] - y
            L[3] = L[3] - y
            L[5] = L[5] - y
            L[7] = L[7] - y
            L[9] = L[9] - y
            
            f.write('/home/cherry/kaki_extracted/test/'+str(i)+' '+str(L[0])+' '+str(L[1])+' '+str(L[2])+' '+str(L[3])+' '+str(L[4])+' '+str(L[5])+' '+str(L[6])+' '+str(L[7])+' '+str(L[8])+' '+str(L[9]))
            f.write('\n')
            

    cv2.imwrite('/home/cherry/kaki_extracted/test/'+i,cropImg)

    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    cv2.imshow("contours", img)
f.close()
cv2.waitKey()
cv2.destroyAllWindows()


