#coding=utf-8

import  xml.dom.minidom
from xml.etree import ElementTree as ET

def getXML(xml_name,arg = 'train'):

#打开xml
    if arg == 'train':
        per = ET.parse('/home/cherry/keras/data/train/xml2/' + xml_name + '.xml')
    else:
        per = ET.parse('/home/cherry/keras/data/test/xml2/' + xml_name + '.xml')     

#得到文档元素对象
#root = dom.documentElement

#cc = root.getElementsByTagName('xmin')

    p=per.findall('./object')
    bp=per.findall('./object/bndbox')

    oneper1 = p[0]
    oneper2 = p[1]
    box1=bp[0]
    box2=bp[1]

    child1 = oneper1.getchildren()
    if child1[0].text == 'head':
       childbox1 = box1.getchildren()
       head_x = childbox1[0]
       head_y = childbox1[1]
    else:
       childbox1 = box2.getchildren()
       head_x = childbox1[0]
       head_y = childbox1[1]
   
    child1 = oneper2.getchildren()

    if child1[0].text == 'c':
       childbox1 = box2.getchildren()
       di_x = childbox1[0]
       di_y = childbox1[1]
    else:
       childbox1 = box1.getchildren()
       di_x = childbox1[0]
       di_y = childbox1[1]
   
    ###print(int(head_x.text),int(head_y.text),int(di_x.text),int(di_y.text))   
    
    return head_x.text,head_y.text,di_x.text,di_y.text

##print(getXML('005'))
     
  #grandchild = child.getchildren()
  #for one in grandchild:
  #print grandchild.text

