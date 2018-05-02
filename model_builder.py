# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 23:29:40 2018

@author: TaitavaBobo§
"""
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50

def create_VGG16_trainable_model(X_train, X_test, y_train, y_test, number_of_classes, learning_rate):
    
    input_shape = X_train.shape[1:]
    
    base_model = VGG16(weights='imagenet', include_top=False, input_shape = input_shape)
    
    w = base_model.output
    w = Flatten()(w)
    w = Dense(1024, activation='relu')(w)
    output = Dense(number_of_classes, activation = 'softmax')(w)
    model = Model(inputs = [base_model.input], outputs = [output])
    
    
    model.layers[-1].trainable = True
    model.layers[-2].trainable = True
    model.layers[-3].trainable = True
    model.layers[-4].trainable = True
    model.layers[-5].trainable = True
    #model.layers[-6].trainable = True
    #model.layers[-7].trainable = True
    
    model.summary()

    optimizer = SGD(lr = learning_rate)
    #optimizer = Adam(lr = 0.01)
    model.compile(optimizer = optimizer,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

    return model


def create_VGG16_freezed_model(X_train, X_test, y_train, y_test, number_of_classes, learning_rate):
    
    input_shape = X_train.shape[1:]
    
    base_model = VGG16(weights='imagenet', include_top=False, input_shape = input_shape)
    
    w = base_model.output
    w = Flatten()(w)
    w = Dense(1024, activation='relu')(w)
    output = Dense(number_of_classes, activation = 'softmax')(w)
    model = Model(inputs = [base_model.input], outputs = [output])
    
    model.layers[-1].trainable = True
    
    #No training for the upper layers to see the performance
    """
    #model.layers[-1].trainable = True
    #model.layers[-2].trainable = True
    #model.layers[-3].trainable = True
    #model.layers[-4].trainable = True
    #model.layers[-5].trainable = True
    #model.layers[-6].trainable = True
    #model.layers[-7].trainable = True
    """
    
    model.summary()

    optimizer = SGD(lr = learning_rate)
    #optimizer = Adam(lr = 0.01)
    model.compile(optimizer = optimizer,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

    return model

def create_VGG19_freezed_model(X_train, X_test, y_train, y_test, number_of_classes, learning_rate):
    
    input_shape = X_train.shape[1:]
    base_model = VGG19(weights='imagenet', include_top=False, input_shape = input_shape)
    
    w = base_model.output
    w = Flatten()(w)
    w = Dense(1024, activation='relu')(w)
    output = Dense(number_of_classes, activation = 'softmax')(w)
    model = Model(inputs = [base_model.input], outputs = [output])
    
    model.layers[-1].trainable = True
    
    #No training for the upper layers to see the performance
    """
    #model.layers[-1].trainable = True
    #model.layers[-2].trainable = True
    #model.layers[-3].trainable = True
    #model.layers[-4].trainable = True
    #model.layers[-5].trainable = True
    #model.layers[-6].trainable = True
    #model.layers[-7].trainable = True
    """
    
    model.summary()

    optimizer = SGD(lr = learning_rate)
    #optimizer = Adam(lr = 0.01)
    model.compile(optimizer = optimizer,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

    return model


if __name__ == '__main__':
    print("Do nothing")