# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:44:22 2018

@author: hakala24
"""

import xml.etree.ElementTree as ET
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image
import sys

def create_fullImage_dict():
    
    #Test or full path
    #images_path = 'RRData/images/*.jpg'
    images_path = 'RRData/images_example/*.jpg'
    
    images_dict = {}
    
    for filename in glob.glob(images_path):
        
        first_split = filename.split('.')
        without_jpg_string = first_split[0]
        image_number_string = without_jpg_string.split('\\')[-1]
        #RRData/images\2319311 for some reason
        
        img = Image.open(filename)
        #numpy_img = np.array(img)
        images_dict[image_number_string] = img
        
    return images_dict

def get_sub_images(images_dict):
    
    images = []
    classes = []
 
    #Full or test path
    #xml_path = 'RRData/annotations/*.xml'
    xml_path = 'RRData/annotations_example/*.xml'
        
    for filename in glob.glob(xml_path):
        
        print(filename)
        #Select image based on xml-file name
        first_split = filename.split('.')
        without_xml_string = first_split[0]
        xml_string = without_xml_string.split('\\')[-1]
        imageString = xml_string.split('-')[0] #Without ending -xxxxxx -part
        img = images_dict[imageString] #Search from previously created dict
        

        #Get objects and bounding boxes from XML-files
        tree = ET.parse(filename)
        
        #node is the text between <object> </object>
        for node in tree.iter('object'):
                     
            #Set 0 for error handling
            xmin = 0
            ymin = 0
            xmax = 0
            ymax = 0
            
            #iterates over all elements in object-node
            for elem in node.iter():
       
                if not elem.tag==node.tag:
                    #Prints all elements of the xml tag
                    #print("{}: {}".format(elem.tag, elem.text))
                    if elem.tag == "name":
                        object_name = elem.text
                    if elem.tag == "xmin":
                        xmin = int(elem.text)
                    if elem.tag == "ymin":
                        ymin = int(elem.text)
                    if elem.tag == "xmax":
                        xmax = int(elem.text)
                    if elem.tag == "ymax":
                        ymax = int(elem.text)
                    
            
            if (xmin + ymin + xmax + ymax) == 0:
                print("No full bounding box on object")
                sys.exit("No full bounding on object")
                
            #Extract subimage from img with bounding box
            else:           
                #The box is a 4-tuple defining the left, upper, right, and lower pixel coordinate.                
                #print("xmin: ", xmin, " ymin: ", ymin, " xmax: ", xmax, " ymax: ", ymax)
                print(xmin,ymax,xmax-xmin,ymax-ymin)
                img_cropped = img.crop((xmin,ymin,xmax,ymax))
                #img_cropped.show()
                images.append(img_cropped)
                classes.append(object_name)

        
    return (images, classes)        

images_dict = create_fullImage_dict()
(images, classes) = get_sub_images(images_dict)