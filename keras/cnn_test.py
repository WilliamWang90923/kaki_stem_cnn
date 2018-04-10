#coding=utf-8
from keras import layers
from keras import optimizers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,Convolution2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image
from keras.models import Sequential
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *
import h5_gener
from keras.layers import Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate  
from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D  
import pydot
import numpy as np  
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import cv2
from keras.optimizers import *
from keras.callbacks import *

testmodel = load_model('/home/cherry/keras/kaki_panasonic.h5')
img_path = '/home/cherry/kaki_extracted/testnew020.png'
img = image.load_img(img_path,grayscale=False,target_size=(96, 96,3))
test_data = np.zeros((1,96,96,3))
test_data[0,:,:,0:3] = img
point = testmodel.predict(test_data)
point = np.squeeze(point)
print(point)
imageresult=cv2.imread(img_path)
res=cv2.resize(imageresult,(96,96),interpolation=cv2.INTER_CUBIC)
cv2.circle(res,(int(point[0]*96),int(point[1]*96)),3,(55,255,155),2)
cv2.circle(res,(int(point[2]*96),int(point[3]*96)),3,(55,255,155),2)
cv2.circle(res,(int(point[4]*96),int(point[5]*96)),3,(55,255,155),2)
cv2.circle(res,(int(point[6]*96),int(point[7]*96)),3,(55,255,155),2)
cv2.circle(res,(int(point[8]*96),int(point[9]*96)),3,(55,255,155),2)
cv2.imshow("result",res)
cv2.imwrite("/home/cherry/result5.png",res)


cv2.waitKey()
cv2.destroyAllWindows()

