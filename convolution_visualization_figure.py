# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 12:33:20 2018

@author: TaitavaBobo§
"""

import numpy as np
import glob
from PIL import Image

def load_data():
    
    globPath = 'C:/Users/TaitavaBobo§/Desktop/SpyderProjects/CatsNDogs/train/train/*.jpg'

    imageData = []
    labels = []
    i = 0
    for imagepath in glob.glob(globPath):
        
        image = Image.open(imagepath)
        image = image.resize((128,128))
        numpyImage = np.array(image)
        imageData.append(numpyImage)
        
        path_split = imagepath.split('\\')[1].split('.')
        labels.append(path_split[0])
        
        i = i+1
        if i==1:
            break
        
    return imageData, labels

if __name__ == '__main__':
    print("esa")