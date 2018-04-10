#coding=utf-8

import  xml.dom.minidom
from xml.etree import ElementTree as ET

def getXML(xml_name):

#打开xml
    
    per = ET.parse('/home/cherry/keras/data/train/xml3/' + xml_name)
   
            

#得到文档元素对象
#root = dom.documentElement

#cc = root.getElementsByTagName('xmin')

    p=per.findall('./object')
    bp=per.findall('./object/bndbox')

    oneper1 = p[0]
    oneper2 = p[1]
    oneper3 = p[2]
    oneper4 = p[3]
    oneper5 = p[4]
    
    box1=bp[0]
    box2=bp[1]
    box3=bp[2]
    box4=bp[3]
    box5=bp[4]
    

    child1 = oneper1.getchildren()
    if child1[0].text == 'point':
       childbox1 = box1.getchildren()
       point_x = childbox1[0]
       point_y = childbox1[1]
    elif child1[0].text == 'ob1':
       childbox1 = box1.getchildren()
       ob1_x = childbox1[0]
       ob1_y = childbox1[1]
    elif child1[0].text == 'ob2':
       childbox1 = box1.getchildren()
       ob2_x = childbox1[0]
       ob2_y = childbox1[1]
    elif child1[0].text == 'ob3':
       childbox1 = box1.getchildren()
       ob3_x = childbox1[0]
       ob3_y = childbox1[1]
    else:
       childbox1 = box1.getchildren()
       ob4_x = childbox1[0]
       ob4_y = childbox1[1]

    child1 = oneper2.getchildren()
    if child1[0].text == 'point':
       childbox1 = box2.getchildren()
       point_x = childbox1[0]
       point_y = childbox1[1]
    elif child1[0].text == 'ob1':
       childbox1 = box2.getchildren()
       ob1_x = childbox1[0]
       ob1_y = childbox1[1]
    elif child1[0].text == 'ob2':
       childbox1 = box2.getchildren()
       ob2_x = childbox1[0]
       ob2_y = childbox1[1]
    elif child1[0].text == 'ob3':
       childbox1 = box2.getchildren()
       ob3_x = childbox1[0]
       ob3_y = childbox1[1]
    else:
       childbox1 = box2.getchildren()
       ob4_x = childbox1[0]
       ob4_y = childbox1[1]

    child1 = oneper3.getchildren()
    if child1[0].text == 'point':
       childbox1 = box3.getchildren()
       point_x = childbox1[0]
       point_y = childbox1[1]
    elif child1[0].text == 'ob1':
       childbox1 = box3.getchildren()
       ob1_x = childbox1[0]
       ob1_y = childbox1[1]
    elif child1[0].text == 'ob2':
       childbox1 = box3.getchildren()
       ob2_x = childbox1[0]
       ob2_y = childbox1[1]
    elif child1[0].text == 'ob3':
       childbox1 = box3.getchildren()
       ob3_x = childbox1[0]
       ob3_y = childbox1[1]
    else:
       childbox1 = box3.getchildren()
       ob4_x = childbox1[0]
       ob4_y = childbox1[1]

    child1 = oneper4.getchildren()
    if child1[0].text == 'point':
       childbox1 = box4.getchildren()
       point_x = childbox1[0]
       point_y = childbox1[1]
    elif child1[0].text == 'ob1':
       childbox1 = box4.getchildren()
       ob1_x = childbox1[0]
       ob1_y = childbox1[1]
    elif child1[0].text == 'ob2':
       childbox1 = box4.getchildren()
       ob2_x = childbox1[0]
       ob2_y = childbox1[1]
    elif child1[0].text == 'ob3':
       childbox1 = box4.getchildren()
       ob3_x = childbox1[0]
       ob3_y = childbox1[1]
    else:
       childbox1 = box4.getchildren()
       ob4_x = childbox1[0]
       ob4_y = childbox1[1]

    child1 = oneper5.getchildren()
    if child1[0].text == 'point':
       childbox1 = box5.getchildren()
       point_x = childbox1[0]
       point_y = childbox1[1]
    elif child1[0].text == 'ob1':
       childbox1 = box5.getchildren()
       ob1_x = childbox1[0]
       ob1_y = childbox1[1]
    elif child1[0].text == 'ob2':
       childbox1 = box5.getchildren()
       ob2_x = childbox1[0]
       ob2_y = childbox1[1]
    elif child1[0].text == 'ob3':
       childbox1 = box5.getchildren()
       ob3_x = childbox1[0]
       ob3_y = childbox1[1]
    else:
       childbox1 = box5.getchildren()
       ob4_x = childbox1[0]
       ob4_y = childbox1[1]
    ###print(int(head_x.text),int(head_y.text),int(di_x.text),int(di_y.text))   
    
    return [int(point_x.text),int(point_y.text),int(ob1_x.text),int(ob1_y.text),int(ob2_x.text),int(ob2_y.text),int(ob3_x.text),int(ob3_y.text),int(ob4_x.text),int(ob4_y.text)]

#print(getXML('003.xml'))
     
  #grandchild = child.getchildren()
  #for one in grandchild:
  #print grandchild.text

