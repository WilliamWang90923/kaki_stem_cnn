#coding=utf-8
from keras import layers
from keras import optimizers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
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
from data.h5_gener import *
from keras.layers import Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate  
from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D  
import pydot
import numpy as np  
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import cv2

seed = 7  
np.random.seed(seed)  

img_path = '/home/cherry/keras/data/train/image2/'
img = image.load_img(img_path+'001.png',target_size=(224,224))
img.save('/home/cherry/test.png')
print('image_size:'+str(img.size))

X_train,Y_train,X_test,Y_test = load_inputdata()

#Y_train = np.squeeze(Y_train,axis=(1,2))
#Y_test = np.squeeze(Y_test,axis=(1,2))

# Reshape


print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
print(Y_test)
  
def Conv2d_BN(x, nb_filter,kernel_size, padding='same',strides=(1,1),name=None):  
    if name is not None:  
        bn_name = name + '_bn'  
        conv_name = name + '_conv'  
    else:  
        bn_name = None  
        conv_name = None  
  
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)  
    x = BatchNormalization(axis=3,name=bn_name)(x)  
    return x  
  
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


X_input = Input((224,224,3))  
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
X_output = X  
    ###model = Model(inputs = X_input,output = X,name='KakiModel')  
    ###return X_output 

KakiModel = Model(inputs = X_input,output = X_output,name='kakiModel') 
optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-8)
KakiModel.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])  
KakiModel.summary()  

KakiModel.fit(X_train, Y_train, shuffle=True,batch_size=8, epochs=1000)
KakiModel.save('/home/cherry/keras/data/kaki_model_2.h5')
'''
testmodel = load_model('/home/cherry/keras/data/kaki_model_2.h5')
testmodel.fit(X_train, Y_train, batch_size=8, epochs=1)

###preds = testmodel.evaluate(X_test, Y_test, batch_size=1)

model_json_str = json.dumps(model.get_config())
open('kaki_model.json','w').write(model_json_str)
model.save_weights("kaki_model_weights.h5",True)
'''
img_path = '/home/cherry/keras/data/test/image2/002.png'
img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x/255.

print(testmodel.predict(x))

### END CODE HERE ###
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
