# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:41:41 2018

@author: TaitavaBobo§
"""
from skimage import data, io, filters
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
import Image_extractor
import class_extractor
from sklearn import model_selection
from keras import regularizers
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import CuDNNLSTM, GRU, LSTM
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD, Adam

def load_data(globPath = 'Images/*.jpg'):

    #globPath = 'Images/*.jpg'
    
    list_not_sorted = []
    for filename in glob.glob(globPath):
        list_not_sorted.append(filename)
    
    list_sorted = sorted(list_not_sorted, key=lambda x: int(x.split("image")[1].split('.')[0]))
    
    filepaths = []
    for file in list_sorted:
        filepaths.append(file)        
    
        
    targets = []
    with open('Classes/classes.csv', 'r') as fp:
        for line in fp:
            #print(line)
            # Otherwise, split the line
            values = line.split(",")
            targets.append(str(values[1].rstrip()))            
    fp.close()
    
    return filepaths, targets


def create_model(X_train, X_test, y_train, y_test):
    

   #Sort according to the number between image'xxxx'.jpg
   list_sorted = sorted(list_not_sorted, key=lambda x: int(x.split("image")[1].split('.')[0]))

   #l1 = Regularizer()

    
   model = Sequential()
    
   model.add(Conv2D(filters = 32, 
                     kernel_size = (5, 5),
                     activation = 'relu',
                     padding = 'same',
                     input_shape=(128,128,3)))    
   model.add(MaxPooling2D(2,2))
   model.add(Dropout(0.25))
   
   model.add(Conv2D(filters = 32, 
                     kernel_size = (5, 5),
                     padding = 'same',
                     activation = 'relu'))
   model.add(MaxPooling2D(2,2))
   model.add(Dropout(0.2))
    
   #model.add(Conv2D(filters = 32, 
   #                  kernel_size = (5, 5),
   #                  padding = 'same',
   #                  activation = 'relu'))
   #model.add(MaxPooling2D(2,2))
   #model.add(Dropout(0.1))
   

   model.add(Flatten())
   model.add(Dense(256, activation = 'relu'))
   model.add(Dense(y_train.shape[1], activation='softmax'))
    
   model.summary()
    
   optimizer = SGD(lr = 1e-1)
   model.compile(optimizer = optimizer,
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    
   return model

def train_model(model, X_train, X_test, y_train, y_test):

    history = model.fit(X_train, y_train, epochs = 150, 
            validation_data = (X_test, y_test))


    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['acc'], 'ro-', label = "Train Accuracy")
    ax[0].plot(history.history['val_acc'], 'go-', label = "Test Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy / %")
    ax[0].legend(loc = "best")
    ax[0].grid('on')
    
    ax[1].plot(history.history['loss'], 'ro-', label = "Train Loss")
    ax[1].plot(history.history['val_loss'], 'go-', label = "Test Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend(loc = "best")
    ax[1].grid('on')
    
    plt.tight_layout()
    plt.savefig("Accuracy.pdf", bbox_inches = "tight")
    

def split_data(imagedata, targets):
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(imagedata, targets, test_size = 0.2, random_state = 1924)
    
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    
    parent_list, dictionaries = class_extractor.return_class_dictionaries()
    
    imagefilepaths, targets = load_data(globPath = 'ResizedImages/*.jpg')    
    
    upper_category_labels = class_extractor.return_upper_category_target_list(targets)
    
    imageData = []
    upper_category_labels_rgb_images = []
    i = 0
    #Check for 3 color channels, or 3 dimensions 128x128x3
    #Remove if image is not RGB image
    for imagepath in imagefilepaths:
        image = io.imread(imagepath)
        numpyImage = np.array(image)
        
        if numpyImage.ndim == 3:
            imageData.append(numpyImage)
            upper_category_labels_rgb_images.append(upper_category_labels[i])            
        i = i + 1
    
    
    temp1 = []
    temp2 = []
    i = 0
    #Remove categories not in parent list
    #Current filterin in class extractor; these categories are not in any subgroup
    for label in upper_category_labels_rgb_images:
        if label in parent_list:            
            temp1.append(imageData[i])
            temp2.append(label)
        i = i + 1
        
    finalData = np.array(temp1)
    categories = temp2
    
    #output = set()
    #for x in upper_category_labels_rgb_images:
    #    output.add(x)
    #print(output)
    
    X_train, X_test, y_train, y_test = split_data(finalData, categories) 
    
    
    lb = LabelBinarizer()
    lb.fit(y_train)
    y_train = lb.transform(y_train)
    y_test = lb.transform(y_test)
    
    
    model = create_model(X_train, X_test, y_train, y_test)
    train_model(model, X_train, X_test, y_train, y_test)


    
