# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 16:21:28 2018

@author: TaitavaBobo§
"""

from collections import Counter
from random import randint
from skimage import data, io, filters
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
import Image_extractor
import class_extractor
import confusion_matrix_implementation
from sklearn import model_selection
from keras import regularizers
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import CuDNNLSTM, GRU, LSTM
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import SGD, Adam
from keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.models import Model


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
        if i<-1:
            break
        
    return imageData, labels



def split_data(imagedata, targets, test_size):
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(imagedata, targets, test_size = test_size)
    
    return X_train, X_test, y_train, y_test

def create_VGG16_model(X_train, X_test, y_train, y_test):
    
    
    base_model = VGG16(weights='imagenet', include_top=False, input_shape = (128,128,3))
    
    w = base_model.output
    w = Flatten()(w)
    w = Dense(2056, activation='relu')(w)
    output = Dense(2, activation = 'softmax')(w)
    model = Model(inputs = [base_model.input], outputs = [output])
    
    model.layers[-5].trainable = True
    model.layers[-6].trainable = True
    model.layers[-7].trainable = True
    
    model.summary()

    optimizer = SGD(lr = 0.0001)
    #optimizer = Adam(lr = 0.01)
    model.compile(optimizer = optimizer,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

    return model


def create_basic_CNN(X_train, y_train, X_test, y_test):
    
    #l1 = Regularizer()
    
   model = Sequential()
    
   model.add(Conv2D(filters = 32, 
                     kernel_size = (5, 5),
                     activation = 'relu',
                     padding = 'same',
                     input_shape=(128, 128, 3)
                     ,W_regularizer=regularizers.l1(0.01)
                     ))    
   model.add(MaxPooling2D(2,2))
   
   model.add(Conv2D(filters = 32, 
                     kernel_size = (5, 5),
                     padding = 'same',
                     activation = 'relu'
                     ,W_regularizer=regularizers.l1(0.01)))
   model.add(MaxPooling2D(2,2))

   model.add(Conv2D(filters = 32, 
                     kernel_size = (5, 5),
                     padding = 'same',
                     activation = 'relu'
                     ,W_regularizer=regularizers.l1(0.01)))
   model.add(MaxPooling2D(2,2))
    
   model.add(Conv2D(filters = 32, 
                     kernel_size = (5, 5),
                     padding = 'same',
                     activation = 'relu'
                     ,W_regularizer=regularizers.l1(0.01)))
   model.add(MaxPooling2D(2,2))

   model.add(Flatten())
   model.add(Dense(512, activation = 'relu'))
   model.add(Dense(2, activation='softmax'))
    
   model.summary()
    
   optimizer = SGD(lr = 1e-5)
   model.compile(optimizer = optimizer,
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

   return model



def train_model(model, X_train, X_test, y_train, y_test, n_epochs = 1, batch_size = 32):

    history = model.fit(X_train, y_train, epochs = n_epochs, batch_size = batch_size,
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
    
    return history

if __name__ == '__main__':
    if 'imageData' not in locals():
        imageData, labels = load_data()
        imageData = np.array(imageData)
    #print("start")
    
    if len(labels) > 10:
        
        lb = LabelBinarizer()
        lb.fit(labels)
        labels_binarized = lb.transform(labels)
        labels_cateogorical = to_categorical(labels_binarized, num_classes=2)
        
        
        #Split data (CrossValidation needed)
        test_size = 0.2
        X_train, X_test, y_train, y_test = split_data(imageData, labels_cateogorical,test_size)
        
        
        #Create model
        #Mmodel = create_basic_CNN(X_train, X_test, y_train, y_test)        
        model = create_VGG16_model(X_train, X_test, y_train, y_test)
        
        
        #Training information
        n_epochs = 5
        batch_size = 32
        history = train_model(model,  X_train, X_test, y_train, y_test, n_epochs, batch_size)
        
        #Confusion matrix
        le = LabelEncoder()
        le.fit(labels)        
        y_test_labels = lb.inverse_transform(y_test)
        y_test_labels_encoded = le.transform(y_test_labels)
        
        numpy_list_labels = np.array(labels)
        unique_labels = np.unique(numpy_list_labels)
        
        #Continuous values
        y_pred = model.predict(X_test)
        #LabelEncoded y_pred
        y_pred = y_pred.argmax(axis=1)
        confusion_matrix_implementation.prepare_confusion_matrix(unique_labels, y_test_labels_encoded, y_pred)

    