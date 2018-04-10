#coding=utf-8
import os
import h5py
import numpy as np
import matplotlib
from keras.preprocessing import image

def load_inputdata():

    RESU_X, RESU_Y =960., 540.

    rootdir = "/home/cherry/keras/data/"

    f=h5py.File("/home/cherry/keras/data/train.h5","w")

    file = open(rootdir + 'train.txt')

    list = os.listdir(rootdir + 'train/image2')

    train_data = np.zeros((len(list),224,224,3))
    label_data = np.zeros((len(list),4))

    lines = file.readlines()

    for line in lines:

        img = image.load_img(rootdir + line[0:20],target_size=(224,224))
        train_data[lines.index(line),:,:,:] = img
        number = line.split(' ')
        label_data[lines.index(line),:] = [int(number[1])/RESU_X,int(number[2])/RESU_Y,int(number[3])/RESU_X,int(number[4])/RESU_Y]
    ####for test
        ##print(label_data[lines.index(line)])
    f.create_dataset('train_x', data = train_data)
    f.create_dataset('label_x', data = label_data)

    train_set_x = h5py.File(rootdir + 'train.h5', "r")
    X_train_orig = np.array(train_set_x['train_x'][:])
    Y_train_orig = np.array(train_set_x['label_x'][:])
    Y_train_orig = Y_train_orig.reshape(len(list),1,1,4)

    f=h5py.File("/home/cherry/keras/data/test.h5","w")

    file = open(rootdir + 'test.txt')

    list = os.listdir(rootdir + 'test/image2')

    train_data = np.zeros((len(list),224,224,3))
    label_data = np.zeros((len(list),4))

    lines = file.readlines()

    for line in lines:

        img = image.load_img(rootdir + line[0:19],target_size=(224,224))
        train_data[lines.index(line),:,:,:] = img
        number = line.split(' ')
        label_data[lines.index(line),:] = [int(number[1])/RESU_X,int(number[2])/RESU_Y,int(number[3])/RESU_X,int(number[4])/RESU_Y]

    f.create_dataset('text_x', data = train_data)
    f.create_dataset('label_y', data = label_data)

    train_set_y = h5py.File(rootdir + 'test.h5', "r")
    X_test_orig = np.array(train_set_y['text_x'][:])
    Y_test_orig = np.array(train_set_y['label_y'][:])
    Y_test_orig = Y_test_orig.reshape(len(list),1,1,4)

    print(Y_test_orig)
    

   

    X_train = X_train_orig/255.
    Y_train = Y_train_orig

    X_test = X_test_orig/255.
    Y_test = Y_test_orig

    #print ("number of training examples = " + str(X_train.shape[0]))
    #print ("number of training examples = " + str(X_test.shape[0]))
    #print ("X_train shape: " + str(X_train.shape))
    #print ("Y_train shape: " + str(Y_train.shape))
    #print ("X_test shape: " + str(X_test.shape))
    #print ("Y_test shape: " + str(Y_test.shape))


    return X_train,Y_train,X_test,Y_test
###X_train shape: (n, 224, 224, 3)
###Y_train shape: (n, 4)





    
