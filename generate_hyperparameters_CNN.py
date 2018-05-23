# -*- coding: utf-8 -*-
"""
Created on Wed May  2 23:01:09 2018

@author: TaitavaBobo§
"""

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import Model
import random
import numpy as np




def generate_CNN(X_train, X_test, y_train, y_test, number_of_classes, hyperparameters):
    
        
    feature_maps = hyperparameters.get("feature maps")
    kernel_size = hyperparameters.get("kernel size")
    number_of_convolutional_layers = hyperparameters.get("convolution layers")
    number_of_dense_layers = hyperparameters.get("dense layers")
    learning_rate  = hyperparameters.get("learning rate")
    
    
    input_shape = X_train.shape[1:]

    model = Sequential()

    model.add(Conv2D(filters = feature_maps,
                    kernel_size = (kernel_size, kernel_size),
                    padding = 'same',
                    activation = 'relu',
                    input_shape=input_shape))
    model.add(MaxPooling2D(2,2))
    
    i = 0
    while i < number_of_convolutional_layers - 1:    
        
        model.add(Conv2D(filters = feature_maps,
                kernel_size = (kernel_size, kernel_size),
                padding = 'same',
                activation = 'relu'))
        model.add(MaxPooling2D(2,2))
        i += 1
    
    model.add(Flatten())
    
    j = 0
    while j < number_of_dense_layers:
        model.add(Dense(128, activation = 'relu'))
        model.add(Dropout(0.25))
        j += 1

    model.add(Dense(number_of_classes, activation='softmax'))

    model.summary()

    optimizer = SGD(lr = learning_rate)
    model.compile(optimizer = optimizer,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])
    
    return model

def generate_hyperparameters():
    
    selected_parameters = {}
    
    dense_layers = [1,2]
    convolution_layers = [1,2,3,4]
    kernel_sizes = [5, 8, 11, 14]
    feature_maps = [16, 32, 48, 64]
    learning_rate = [10**-6, 10**-2]
    input_size = [32, 64, 96, 128]
    
    """
    dense_layers = [2]
    convolution_layers = [4]
    kernel_sizes = [14]
    feature_maps = [64]
    learning_rate = [10**-6, 10**-2]
    input_size = [96]
    """
    dense_layers = random.choice(dense_layers)
    convolution_layers = random.choice(convolution_layers)
    kernel_sizes = random.choice(kernel_sizes)
    feature_maps = random.choice(feature_maps)
    input_size = random.choice(input_size)
    learning_rate = random.uniform(learning_rate[0], learning_rate[1])
    
    selected_parameters["dense layers"] = dense_layers
    selected_parameters["convolution layers"] = convolution_layers
    selected_parameters["kernel size"] = kernel_sizes
    selected_parameters["feature maps"] = feature_maps
    selected_parameters["input size"] = input_size
    selected_parameters["learning rate"] = learning_rate    
    
    return selected_parameters

if __name__ == '__main__':
    print("Nothing")
    