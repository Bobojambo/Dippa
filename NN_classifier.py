# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:41:41 2018

@author: TaitavaBobo§
"""

import numpy as np
import glob
import os
from PIL import Image

def load_data():

    #x_test = np.load("X_test.npy")
    #data = np.load("X_train.npy")
    data = []
    targets = []
    with open('Classes/classes.csv', 'r') as fp:
        for line in fp:
            #print(line)
            # Otherwise, split the line
            values = line.split(",")
            targets.append(str(values[1].rstrip()))            
    fp.close()
    
    list_not_sorted = []
    for filename in glob.glob('Images/*.jpg'):
        list_not_sorted.append(filename)
    
    #Sort according to the number between image'xxxx'.jpg
    list_sorted = sorted(list_not_sorted, key=lambda x: int(x.split("image")[1].split('.')[0]))
    
    filepaths = []
    data = []
    for file in list_sorted:
        filepaths.append(file)
        #image = Image.open(file)
    #for filenumber in list_sorted:
    #    path = os.path.join('Images/image', filenumber, '.jpg')
    #    filepaths.append(path)
        
    
    return filepaths, targets


if __name__ == '__main__':
    filepaths, targets = load_data() # this calls main function
    image = Image.open(filepaths[0])
    image.show()
    
    np_array = np.array(image)
    
    #testi = not_sorted[0].split("image")[1].split('.')[0]