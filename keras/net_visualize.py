#coding=utf-8
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,Convolution2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *
from data.h5_gener import *
from keras.layers import Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate  
from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D  
import pydot
import pylab
import numpy as np  
import keras.backend as K
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import cv2

seed = 1024  
np.random.seed(seed)  

img_path = '/home/cherry/keras/data/train/image1/'
img = image.load_img(img_path+'001.png',target_size=(224,224))
plt.imshow(img)
pylab.show()
print(img.size)
kaki = image.img_to_array(img)
print (kaki.shape)


def visualize_kaki(conv_kaki):
    kaki = np.squeeze(conv_kaki,axis=0)
    print (kaki.shape)
    plt.imshow(kaki)
    pylab.show()

    
"""def Conv2d_BN(x, nb_filter,kernel_size, padding='same',strides=(1,1),name=None):  
    if name is not None:  
        bn_name = name + '_bn'  
        conv_name = name + '_conv'  
    else:  
        bn_name = None  
        conv_name = None  
  
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)  
    x = BatchNormalization(axis=3,name=bn_name)(x)  
    return x  """

  
def Inception(x,nb_filter):  
    branch1x1 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)  
  
    branch3x3 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)  
    branch3x3 = Conv2d_BN(branch3x3,nb_filter,(3,3), padding='same',strides=(1,1),name=None)  
  
    branch5x5 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)  
    branch5x5 = Conv2d_BN(branch5x5,nb_filter,(1,1), padding='same',strides=(1,1),name=None)  
  
    branchpool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)  
    branchpool = Conv2d_BN(branchpool,nb_filter,(1,1),padding='same',strides=(1,1),name=None)  
  
    x = concatenate([branch1x1,branch3x3,branch5x5,branchpool],axis=3)  
  
    return x  

def KakiModel(input_shape=(224,224,3)):
    X_input = Input(input_shape)  
      #padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))  
    X = Conv2d_BN(X_input,64,(7,7),strides=(2,2),padding='same')  
    X = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(X) 
    X = Conv2d_BN(X,192,(3,3),strides=(1,1),padding='same')  
    X = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(X) 
    X = Inception(X,64)#256  
    X = Inception(X,120)#480  
    X = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(X) 
    X = Inception(X,128)#512  
    X = Inception(X,128)  
    X = Inception(X,128)  
    X = Inception(X,132)#528  
    X = Inception(X,208)#832  
    X = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(X)  
    X = Inception(X,208)  
    X = Inception(X,256)#1024  
    X = AveragePooling2D(pool_size=(7,7),strides=(7,7),padding='same')(X)  
    X = Dropout(0.5)(X)  
    X = Dense(1000,activation='relu')(X)  
    X = Dense(4,activation='softmax')(X)  
    model = Model(inputs = X_input,output = X,name='KakiModel')  
    return model

model = Sequential()
"""model.add(Conv2D(64,(7,7),strides=(2,2),padding='same',input_shape=(224,224,3))) 
model.add(BatchNormalization(axis=3))
kaki_batch = np.expand_dims(kaki,axis=0)"""

model.add(Convolution2D(3,3,3,input_shape=(224,224,3)))
model.add(Activation('sigmoid'))

kaki_batch = np.expand_dims(kaki,axis=0)
conv_kaki = model.predict(kaki_batch)
visualize_kaki(kaki_batch)

