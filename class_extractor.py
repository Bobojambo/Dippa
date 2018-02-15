# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:25:07 2018

@author: hakala24
"""

import numpy as np
import glob
import xml.etree.ElementTree as ET

def return_values(parent, child):
    
    for value in child.attrib.values():
        #print(value)
        for key in parent.attrib.values():
            #print(key)   
            
            return key, value

def get_classes():
    
    child_3_to_2_dict = {}
    child_3_to_1_dict = {}
    child_1_to_2_dict = {}
    child_1_to_3_dict = {}
    child_2_to_1_dict = {}
    child_2_to_3_dict = {}
    #Full or test path
    #xml_path = 'RRData/annotations/*.xml'
    xml_path = 'RRData/classes.xml'
        
    for filename in glob.glob(xml_path):
        
        print(filename)
        
        tree = ET.parse(filename)        
        root = tree.getroot()
        for child in root:
            if child.tag == "class":
                print(child.tag, child.attrib)
                for child2 in child:
                    
                    if child2.tag == "class":     
                        
                        key, value = return_values(child, child2)
                        
                        if not key in child_1_to_2_dict:
                            child_1_to_2_dict[key] = [value]
                        else:                        
                            child_1_to_2_dict[key].append(value)     
                            
                        for child3 in child2:
                            
                            if child3.tag == "class":
                                
                                key, value = return_values(child2, child3)
                            
                                if not key in child_2_to_3_dict:
                                    child_2_to_3_dict[key] = [value]
                                else:                        
                                    child_2_to_3_dict[key].append(value) 
                                    
                                #Still more childs to check
                                for child4 in child3:
                                    print(child4.attrib)
    
    #.attrib = dict
    print(" ")
    
    return child_1_to_2_dict, child_2_to_3_dict
            

child_1_to_2_dict, child_2_to_3_dict = get_classes()
            
