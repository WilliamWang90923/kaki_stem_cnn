#coding=utf-8
import os
import glob
import re

filelist = glob.glob( '/home/cherry/图片/preview/*.png')
i = 1

for file in filelist:
    os.rename(file, '%03.f.png'%i)
    print "renmame:" + file + " to " + "%03.f.png"%i
    i = i + 1
