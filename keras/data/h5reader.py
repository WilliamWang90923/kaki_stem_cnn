#coding=utf-8
import h5py  
import numpy as np 

f = h5py.File('/home/cherry/keras/data/test.h5','r')   #打开h5文件  
f.keys()                            #可以查看所有的主键  
a = f['label_y'][:]   
print(a)                 #取出主键为data的所有的键值  
f.close()   
