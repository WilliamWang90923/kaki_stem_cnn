#coding=utf-8
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import kakirotation
import xmlreader_test
import cv2

list = os.listdir('/home/cherry/kaki_extracted')
f=file('/home/cherry/kaki_extracted/augmented/label.txt', 'w')
for i in list:

    if os.path.splitext(i)[1] == '.png':     
        image = cv2.imread('/home/cherry/kaki_extracted/' + i)
        img_x = float(image.shape[0])
        img_y = float(image.shape[1])
        for arg in [0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345]:
            points = xmlreader_test.getXML(os.path.splitext(i)[0]+'.xml')
            rotated,pointresult=kakirotation.rotate(image,arg,points)    
            cv2.imwrite('/home/cherry/kaki_extracted/augmented/'+os.path.splitext(i)[0]+str(arg)+'.png',rotated)
            f.write('/home/cherry/kaki_extracted/augmented/'+os.path.splitext(i)[0]+str(arg)+'.png'+' '+
str(pointresult[0][0]/img_x)+' '+str(pointresult[0][1]/img_y)+' '+str(pointresult[1][0]/img_x)+' '+str(pointresult[1][1]/img_y)+' '+str(pointresult[2][0]/img_x)+' '+str(pointresult[2][1]/img_y)+'\n')
    
