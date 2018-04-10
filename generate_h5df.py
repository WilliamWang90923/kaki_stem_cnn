# coding: utf-8  
keras_root = '/home/cherry/keras/'  
import sys  
sys.path.insert(0, caffe_root + 'python')  
import os  
import cv2  
import numpy as np  
import h5py  
from common import shuffle_in_unison_scary, processImage  
import matplotlib.pyplot as plt  
      
def readdata(filepath):  
    fr=open(filepath,'r')  
    filesplit=[]  
    for line in fr.readlines():  
        s=line.split()  
        s[1:]=[float(x) for x in s[1:]]  
        filesplit.append(s)  
    fr.close()  
    return  filesplit  
#因为我们的训练数据可能不是正方形，然而网络的输入的大小是正方形图片，为了避免强制resize引起的图片扭曲，所以我们采用填充的方法  
def sqrtimg(img):  
    height,width=img.shape[:2]  
    maxlenght=max(height,width)  
    sqrtimg0=np.zeros((maxlenght,maxlenght,3),dtype='uint8')  
      
    sqrtimg0[(maxlenght*.5-height*.5):(maxlenght*.5+height*.5),(maxlenght*.5-width*.5):(maxlenght*.5+width*.5)]=img  
    return  sqrtimg0  
      
      
def generate_hdf5():  
      
    labelfile =readdata('../data/my_alige_landmark.txt')  
    F_imgs = []  
    F_landmarks = []  
      
      
    for i,l in enumerate(labelfile):  
        imgpath='../data/'+l[0]  
      
        img=cv2.imread(imgpath)  
        maxx=max(img.shape[0],img.shape[1])  
        img=sqrtimg(img)#把输入图片填充成正方形，因为我们要训练的图片的大小是正方形的图片255*255  
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#图片转为灰度图像  
        f_face=cv2.resize(img,(39,39))#把图片缩放成255＊255的图片  
        # F  
        plt.imshow(f_face,cmap='gray')  
      
      
        f_face = f_face.reshape((1, 39, 39))  
        f_landmark =np.asarray(l[1:],dtype='float')  
   
        F_imgs.append(f_face)  
      
      
        #归一化人脸特征点标签,因为上面height等于width，这里比较懒，直接简写  
        f_landmark=f_landmark/maxx #归一化到0～１之间  
        print f_landmark  
        F_landmarks.append(f_landmark)  
      
      
    F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)  
      
      
    F_imgs = processImage(F_imgs)#图片预处理，包含均值归一化，方差归一化等  
    shuffle_in_unison_scary(F_imgs, F_landmarks)#打乱数据  
      
    #生成h5py格式  
    with h5py.File(os.getcwd()+ '/train_data.h5', 'w') as f:  
    f['data'] = F_imgs.astype(np.float32)  
        f['landmark'] = F_landmarks.astype(np.float32)  
    #因为caffe的输入h5py不是直接使用上面的数据，而是需要调用.txt格式的文件  
    with open(os.getcwd() + '/train.txt', 'w') as f:  
        f.write(os.getcwd() + '/train_data.h5\n')  
    print i  
      
if __name__ == '__main__':  
    generate_hdf5()  
