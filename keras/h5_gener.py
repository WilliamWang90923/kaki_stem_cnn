#coding=utf-8
import os
import h5py
import numpy as np
import matplotlib
from keras.preprocessing import image

def point_arrange(points):
   
    a = [points[2]*points[2]+points[3]*points[3],points[4]*points[4]+points[5]*points[5],points[6]*points[6]+points[7]*points[7],points[8]*points[8]+points[9]*points[9]]
    for i in range(3):
        for j in range(3-i):
            if a[j] > a[j+1]:
                points[2*j+2], points[2*j+3],points[2*j+4],points[2*j+5],a[j], a[j+1] = points[2*j+4], points[2*j+5],points[2*j+2],points[2*j+3],a[j+1], a[j]
    return points

def load_inputdata():

    RESU_X, RESU_Y =960., 540.

    rootdir = "/home/cherry/kaki_extracted/"

    f = h5py.File("/home/cherry/kaki_extracted/trainnew.h5","w") 
    f2=h5py.File("/home/cherry/kaki_extracted/testnew.h5","w")

    file = open(rootdir + 'labelnew2.txt')

    list = os.listdir(rootdir + 'augmented2')

    train_data = np.zeros((4000,96,96,3))
    test_data = np.zeros((800,96,96,3))
    label_data_train = np.zeros((4000,10))
    label_data_test = np.zeros((800,10))

    lines = file.readlines()

    for line in lines:
        if lines.index(line) < 4000:
            img = image.load_img(line.split()[0],grayscale=False,target_size=(96,96))
            train_data[lines.index(line),:,:,0:3] = img            
            number = line.split()
            label_data_train[lines.index(line),:] = point_arrange([float(number[1]),float(number[2]),float(number[3]),float(number[4]),float(number[5]),float(number[6]),float(number[7]),float(number[8]),float(number[9]),float(number[10])])
        else:
            img = image.load_img(line.split()[0],grayscale=False,target_size=(96,96))
            test_data[lines.index(line)-4000,:,:,0:3] = img         
            number = line.split()
            label_data_test[lines.index(line)-4000,:] = point_arrange([float(number[1]),float(number[2]),float(number[3]),float(number[4]),float(number[5]),float(number[6]),float(number[7]),float(number[8]),float(number[9]),float(number[10])])
    ####for test
    print(label_data_test[12])
    print(label_data_test[13])
    print(label_data_test[14])
    print(label_data_test[15])
    f.create_dataset('train_data', data = train_data)
    f.create_dataset('label_data_train', data = label_data_train)
    f2.create_dataset('test_data', data = test_data)
    f2.create_dataset('label_data_test', data = label_data_test)

    '''
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
    '''    

   

   

  #  X_test = X_test_orig/255.
  #  Y_test = Y_test_orig

    


    return X_train,Y_train
###X_train shape: (n, 224, 224, 3)
###Y_train shape: (n, 4)

if __name__ == '__main__':
    load_inputdata()   
    train_set_x = h5py.File('/home/cherry/kaki_extracted/trainnew.h5', "r")
    X_train = np.array(train_set_x['train_data'][:])
    Y_train = np.array(train_set_x['label_data_train'][:])
    #Y_train = Y_train.reshape(len(list),10)
    print ("number of training examples = " + str(X_train.shape[0]))
    #print ("number of training examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    #print ("X_test shape: " + str(X_test.shape))
    #print ("Y_test shape: " + str(Y_test.shape))




    
