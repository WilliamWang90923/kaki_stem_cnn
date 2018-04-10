#coding=utf-8
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
import os
import kakirotation
import cv2

#pic_list = os.listdir('/home/cherry/kaki_extracted/test/')
f1 = open('/home/cherry/kaki_extracted/labelnew.txt')

lines = f1.readlines()

f2=open('/home/cherry/kaki_extracted/labelnew2.txt','w')


for line in lines:

        img = cv2.imread(line.split()[0])
      
        points = [[int(line.split()[1]),int(line.split()[2])],[int(line.split()[3]),int(line.split()[4])],[int(line.split()[5]),int(line.split()[6])],[int(line.split()[7]),int(line.split()[8])],[int(line.split()[9]),int(line.split()[10])]]
        img_y = float(img.shape[0])
        img_x = float(img.shape[1])
        for arg in [0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345]:
            rotated,pointresult=kakirotation.rotate(img,arg,points)    
            cv2.imwrite('/home/cherry/kaki_extracted/augmented2/'+line.split()[0][33:36]+str(arg)+'.png',rotated)
            f2.write('/home/cherry/kaki_extracted/augmented2/'+line.split()[0][33:36]+str(arg)+'.png'+' '+
str(pointresult[0][0]/img_x)+' '+str(pointresult[0][1]/img_y)+' '+str(pointresult[1][0]/img_x)+' '+str(pointresult[1][1]/img_y)+' '+str(pointresult[2][0]/img_x)+' '+str(pointresult[2][1]/img_y)+' '+str(pointresult[3][0]/img_x)+' '+str(pointresult[3][1]/img_y)+' '+str(pointresult[4][0]/img_x)+' '+str(pointresult[4][1]/img_y)+'\n')

#for i in list:
#
#    if os.path.splitext(i)[1] == '.png':     
#        image = cv2.imread('/home/cherry/kaki_extracted/' + i)
#        img_x = float(image.shape[0])
#        img_y = float(image.shape[1])
#        for arg in [0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345]:
#            points = xmlreader_test.getXML(os.path.splitext(i)[0]+'.xml')
#            rotated,pointresult=kakirotation.rotate(image,arg,points)    
#            cv2.imwrite('/home/cherry/kaki_extracted/augmented/'+os.path.splitext(i)[0]+str(arg)+'.png',rotated)
#            f.write('/home/cherry/kaki_extracted/augmented/'+os.path.splitext(i)[0]+str(arg)+'.png'+' '+
#str(pointresult[0][0]/img_x)+' '+str(pointresult[0][1]/img_y)+' '+str(pointresult[1][0]/img_x)+' '+str(pointresult[1][1]/img_y)+' '+str(pointresult[2][0]/img_x)+' '+str(pointresult[2][1]/img_y)+' '+str(pointresult[3][0]/img_x)+' '+str(pointresult[3][1]/img_y)+' '+str(pointresult[4][0]/img_x)+' '+str(pointresult[4][1]/img_y)+'\n')
    
