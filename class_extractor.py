# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:25:07 2018

@author: hakala24
"""

import glob
import xml.etree.ElementTree as ET

#Sets the child as key and parent as value
def append_key_value_pair(parent, child, dictionary):
    for key in child.attrib.values():
        for value in parent.attrib.values():
            
            if not key in dictionary:
                dictionary[key] = [value]
            else:                        
                dictionary[key].append(value)  

            return dictionary
        

def return_values(parent, child):
    
    for value in child.attrib.values():
        #print(value)
        for key in parent.attrib.values():
            #print(key)   
            
            return key, value
        
def return_class_dictionaries():
    
   
    parent_list = []    
    child_2_to_1_dict = {}
    child_3_to_2_dict = {}
    child_4_to_3_dict = {}
    child_5_to_4_dict = {}
    #Full or test path
    #xml_path = 'RRData/annotations/*.xml'
    xml_path = 'RRData/classes.xml'
        
    for filename in glob.glob(xml_path):
        
        print(filename)
        
        tree = ET.parse(filename)        
        root = tree.getroot()
        for child in root:
            if child.tag == "class":
                arvo_avain = child.attrib.values()
                for value in arvo_avain:
                    parent_list.append(value)
                #print(child.tag, child.attrib)
                for child2 in child:
                    if child2.tag == "class":  
                        child_2_to_1_dict = append_key_value_pair(child, child2, child_2_to_1_dict)     
                        
                        for child3 in child2:                            
                            if child3.tag == "class":
                                child_3_to_2_dict = append_key_value_pair(child2, child3, child_3_to_2_dict)       
                                
                                for child4 in child3:                                    
                                    if child4.tag == "class":  
                                        child_4_to_3_dict = append_key_value_pair(child3, child4, child_4_to_3_dict) 
                     
                                        for child5 in child4:                                            
                                            if child5.tag == "class":
                                                child_5_to_4_dict = append_key_value_pair(child4, child5, child_5_to_4_dict) 

    print(" ")
    
    dictionaries_list =  []
    dictionaries_list.append(child_2_to_1_dict)
    dictionaries_list.append(child_3_to_2_dict)
    dictionaries_list.append(child_4_to_3_dict)
    dictionaries_list.append(child_5_to_4_dict)
    
    return parent_list, dictionaries_list

def check_bad_category(category, parent_list, dictionaries):
    category_exists = False
    for parent_category in parent_list:
        if category == parent_category:
            category_exists = True
            return category_exists
    
    for dictionary in dictionaries:
        if category in dictionary:
            category_exists = True
            return category_exists
    
    return category_exists

def return_upper_category_target_list(original_targets_categories, vessel_classification = False):
    
    parent_list, dictionaries = return_class_dictionaries()
    
    child_2_to_1_dict = dictionaries[0]
    child_3_to_2_dict = dictionaries[1]
    child_4_to_3_dict = dictionaries[2]
    child_5_to_4_dict = dictionaries[3]
    
    vessel_labels = []
    target_labels = []
    pre_category = "Empty"
    class_exists = bool
    
    for category in original_targets_categories:
        #.join because the return values are lists, even though they shouldnt be
        return_value = child_5_to_4_dict.get(category)
        if return_value is not None:
            pre_category = category
            category = ''.join(return_value)
        
        return_value = child_4_to_3_dict.get(category)
        if return_value is not None:
            pre_category = category
            category = ''.join(return_value)
        
        return_value = child_3_to_2_dict.get(category)
        if return_value is not None:
            pre_category = category
            category = ''.join(return_value)
            
        return_value = child_2_to_1_dict.get(category)
        if return_value is not None:
            pre_category = category
            category = ''.join(return_value)
        
        #Check if the category is not in the class hierarchy
        class_exists_in_hierarchy = check_bad_category(category, parent_list, dictionaries)
        if class_exists_in_hierarchy is False:
            category = "Empty"

        #For vessel type classification
        if category == "Vessel":
            vessel_labels.append(pre_category)
        else:
            vessel_labels.append("Empty")
        
        target_labels.append(category)
        pre_category = "Empty"
    
    if vessel_classification is True:
        return vessel_labels
    else:
        return target_labels

if __name__ == "__main__":
    parent_list, dictionaries_list = return_class_dictionaries()
