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
import class_extractor
import os
import shutil
import csv

def create_fullImage_dict():
    
    #Test or full path
    images_path = 'RRData/images/*.jpg'
    #images_path = 'RRData/images_example/*.jpg'
    
    images_dict = {}
    
    for filename in glob.glob(images_path):
        
        first_split = filename.split('.')
        without_jpg_string = first_split[0]
        image_number_string = without_jpg_string.split('\\')[-1]
        #RRData/images\2319311 for some reason   
        images_dict[image_number_string] = filename       
        
    return images_dict

def get_sub_images(images_dict, resize_argument, test_images_argument):
    
    images = []
    classes = []
 
    #Full or test path
    if test_images_argument == "Y":
        xml_path = 'RRData/annotations_example/*.xml'
    else:
        xml_path = 'RRData/annotations/*.xml'
        
    
        
    for filename in glob.glob(xml_path):
        
        #print(filename)
        #Select image based on xml-file name
        first_split = filename.split('.')
        without_xml_string = first_split[0]
        xml_string = without_xml_string.split('\\')[-1]
        imageString = xml_string.split('-')[0] #Without ending -xxxxxx -part
        img_filename = images_dict[imageString] #Search from previously created dict
        img = Image.open(img_filename)

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
                #Resize the cropped area to square
                if resize_argument == "True":
                #print("x_max, x_min, x_distance: ", xmax," ", xmin , " ", xmax-xmin)
                #print("y_max, y_min, y_distance: ", ymax," ", ymin , " ", ymax-ymin)
                    y_gap = ymax - ymin
                    x_gap = xmax - xmin
                    if x_gap > y_gap:
                        pixels_to_add = (x_gap - y_gap)/2
                        ymin = ymin - pixels_to_add
                        ymax = ymax + pixels_to_add
                    else:
                        pixels_to_add = (y_gap - x_gap)/2
                        xmin = xmin - pixels_to_add
                        xmax = xmax + pixels_to_add     
                        
                    img_cropped = img.crop((xmin,ymin,xmax,ymax))
                    images.append(img_cropped)
                    classes.append(object_name)
                    
                else:
                    #The box is a 4-tuple defining the left, upper, right, and lower pixel coordinate.                
                    #print("xmin: ", xmin, " ymin: ", ymin, " xmax: ", xmax, " ymax: ", ymax)
                    #print(xmin,ymax,xmax-xmin,ymax-ymin)
                    img_cropped = img.crop((xmin,ymin,xmax,ymax))
                    images.append(img_cropped)
                    classes.append(object_name)         
        #break for single image testing
        #break

    return images, classes

def save_images_to_folder(images, path=None):
    
    if path == None:
        path = "Images/"
        
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    
    
    index = 0
    for image in images:
        #image.show()
        filename = "{}image{}.jpg".format(path,index)
        image.save(filename)
        index += 1
        #if index == len(images):
        #    break
    return

def create_resized_images(images, width=128, height=128):
    reSizedImages = []
    for image in images:
        reSizedImage = image.resize((width, height))
        reSizedImages.append(reSizedImage)
    
    save_images_to_folder(reSizedImages, path="ResizedImages64x64/")
    
    return 

def create_classes_csv(classes):

    path = "Classes/"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    
    with open("Classes/classes.csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        index = 0
        for line in classes:
            writer.writerow([index, line])   
            index += 1
    
    return

if __name__ == "__main__":
    
    input_var = input("Square subimage extraction (True/False): ")    
    argument = str(input_var)
    input_var = str(input("Test images run (Y/N): "))
    test_images_argument = input_var
    
    if argument == "True" or argument == "False":
        images_dict = create_fullImage_dict()
        
        images, classes = get_sub_images(images_dict, argument, test_images_argument)    
        
        parent_list, dictionaries_list = class_extractor.return_class_dictionaries()
        
        save_images_to_folder(images)
        create_resized_images(images)
        create_classes_csv(classes)
    else:
        print("Wrong input argument")