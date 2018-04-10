# coding: utf-8  
import os  
import numpy as np  
from matplotlib import pyplot as plt  
import cv2  
import shutil,string
import xml.dom.minidom
from xmlreader_test import *

rootPath = '/home/cherry/keras/data/'

f = open(rootPath+'train.txt','w')
for i in range(2,3):
    path = 'train/image' + str(i) + '/'
    path2 = 'train/xml' + str(i)
    lists = os.listdir(rootPath + path)
    for listfile in lists:
        if listfile != 'Thumbs.db':
           temp = getXML(listfile.split('.')[0]) 
           f.writelines([path,listfile,' ',temp[0],' ',temp[1],' ',temp[2],' ',temp[3],'\n'])
f.close()

f = open(rootPath+'test.txt','w')
for i in range(2,3):
    path = 'test/image' + str(i) + '/'
    path2 = 'test/xml' + str(i)
    lists = os.listdir(rootPath + path)
    for listfile in lists:
        if listfile != 'Thumbs.db':
           temp = getXML(listfile.split('.')[0],'test') 
           f.writelines([path,listfile,' ',temp[0],' ',temp[1],' ',temp[2],' ',temp[3],'\n'])
f.close()
