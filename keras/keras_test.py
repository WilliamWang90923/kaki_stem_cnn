#coding=utf-8
import pydot
import numpy as np  
from keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import pylab
import cv2

img_path = '/home/cherry/kaki_extracted/test/001.png'
im = Image.open(img_path).convert('L')
plt.imshow(im)
pylab.show()
img = image.load_img(img_path,True,target_size=(96, 96))
x = image.img_to_array(img)
x = x/255.0
img.save('/home/cherry/kaki_extracted/test/gray001.png')
#plt.imshow(img)
#pylab.show()
print(img.mode)
print(np.size(img))
print(np.size(x))
