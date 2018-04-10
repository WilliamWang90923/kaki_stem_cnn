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

seed = 7  
np.random.seed(seed)  



train_set_x = h5py.File('/home/cherry/kaki_extracted/trainnew.h5', "r")
X_train = np.array(train_set_x['train_data'][:])
Y_train = np.array(train_set_x['label_data_train'][:])
test_set_x = h5py.File('/home/cherry/kaki_extracted/testnew.h5', "r")
X_test = np.array(test_set_x['test_data'][:])
Y_test = np.array(test_set_x['label_data_test'][:])   
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of training examples = " + str(Y_train.shape[0]))
print ("number of testing examples = " + str(X_test.shape[0]))
print ("number of testing examples = " + str(Y_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))

#X_train,Y_train = h5_gener.load_inputdata()

#X_train = X_train/255.0
#Y_train = Y_train/255.0
#Y_train = np.squeeze(Y_train,axis=(1,2))
#Y_test = np.squeeze(Y_test,axis=(1,2))

# Reshape

model = Sequential()

model.add(BatchNormalization(input_shape=(96, 96, 3)))

model.add(Convolution2D(24, 5, 5, border_mode='same',init='he_normal', input_shape=(96, 96, 3),dim_ordering='tf'))

model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),border_mode='valid'))

model.add(Convolution2D(36, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),border_mode='valid'))

model.add(Convolution2D(48, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),border_mode='valid'))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),border_mode='valid'))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))

model.add(GlobalAveragePooling2D());

model.add(Dense(200, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(10))
'''
def CNN_model(size):
    input_data = Input(size)
    
    x = Conv2D(24, (5, 5), activation='relu')(input_data)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(96, (2, 2), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
        
    x = Conv2D(128, (2, 2), activation='relu')(x)    

    x = Flatten()(x)
    x = Dense(256,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10,activation='relu')(x)
    output = x
    return Model(inputs=input_data, outputs=output, name='Discriminator')
'''
#model=CNN_model((96, 96, 3))
#optimizer=SGD(lr=0.03, momentum=0.9, nesterov=True)
model.compile(optimizer='rmsprop',loss='mse', metrics=['accuracy'])
epoch_num = 2000
checkpointer = ModelCheckpoint(filepath='kaki_panasonic.h5',verbose=1, save_best_only=True)
#learning_rate = np.linspace(0.03, 0.01, epoch_num)
#change_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
#early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
#check_point= ModelCheckpoint('CNN_model_final.h5', monitor='val_loss',verbose=0,save_best_only=True,save_weights_only=False, mode='auto', period=1)

#model.compile(optimizer=’rmsprop’, loss=’mse’, metrics=[‘accuracy’])

#checkpointer = ModelCheckpoint(filepath=’face_model.h5',verbose=1, save_best_only=True)

#epochs = 300

model.fit(X_train, Y_train, validation_split=0.2,shuffle=True, epochs=epoch_num, batch_size=128, callbacks=[checkpointer], verbose=1,validation_data=[X_test,Y_test])
model.save('my_panasonic.h5')

#KakiModel = Model(inputs = X_input,output = X_output,name='kakiModel') 
#optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-8)
#KakiModel.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])  
#KakiModel.summary()  

#KakiModel.fit(X_train, Y_train, shuffle=True,batch_size=8, epochs=100)
#KakiModel.save('/home/cherry/keras/data/kaki_model_new001.h5')

##testmodel = load_model('/home/cherry/keras/CNN_model_final.h5')
#testmodel.fit(X_train, Y_train, batch_size=8, epochs=1)

###preds = testmodel.evaluate(X_test, Y_test, batch_size=1)

img_path = '/home/cherry/kaki_extracted/test/002.png'
img = image.load_img(img_path,grayscale=True,target_size=(96, 96))
test_data = np.zeros((1,96,96,3))
test_data[0,:,:,0] = img


test_data = test_data/255.0
print(test_data.shape)

print(testmodel.predict(test_data)*255)
'''
### END CODE HERE ###
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
'''
