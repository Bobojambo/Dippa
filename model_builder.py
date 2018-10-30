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
from keras.applications.mobilenet import MobileNet

def return_the_optimizer(learning_rate):
    
    selected_optimizer = SGD(lr = learning_rate)
    
    return selected_optimizer

def create_VGG16_trainable_model(X_train, X_test, y_train, y_test, number_of_classes, learning_rate):
    
    input_shape = X_train.shape[1:]
    
    base_model = VGG16(weights='imagenet', include_top=False, input_shape = input_shape)
    
    w = base_model.output
    w = Flatten()(w)
    w = Dense(256, activation='relu')(w)
    w = Dropout(0.25)(w)
    output = Dense(number_of_classes, activation = 'softmax')(w)
    model = Model(inputs = [base_model.input], outputs = [output])
    
    
    model.layers[-1].trainable = True
    model.layers[-2].trainable = True
    model.layers[-3].trainable = True
    model.layers[-4].trainable = True
    model.layers[-5].trainable = True
    
    model.summary()

    optimizer = SGD(lr = learning_rate)
    model.compile(optimizer = optimizer,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

    return model

def create_VGG16_trainable_model2(X_train, X_test, y_train, y_test, number_of_classes, learning_rate):
    
    model = Sequential()
    
    input_shape = X_train.shape[1:]    
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape = input_shape)
    
    #for layer in vgg16.layers[:-4]:
    #    layer.trainable = False
        
    #for layer in vgg16.layers:
    #    print(layer, layer.trainable)
        
    model.add(vgg16)
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, activation='softmax'))
    
    model.summary()

    optimizer = SGD(lr = learning_rate)
    model.compile(optimizer = optimizer,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

    return model


    
    input_shape = X_train.shape[1:]    
    ResNet = ResNet50(weights='imagenet', include_top=False, input_shape = input_shape)    

    
    model.add(ResNet)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(number_of_classes, activation='softmax'))
    
    model.summary()

    optimizer = SGD(lr = learning_rate)
    model.compile(optimizer = optimizer,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

def create_ResNet_trainable_model(X_train, X_test, y_train, y_test, number_of_classes, learning_rate):
    
    model = Sequential()
    
    input_shape = X_train.shape[1:]    
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape = input_shape)  
    
    for layer in base_model.layers[:-10]:
        layer.trainable = False
        
    for layer in model.layers:
        print(layer, layer.trainable)
    
    w = base_model.output
    w = Flatten()(w)
    w = Dense(512, activation='relu')(w)
    output = Dense(number_of_classes, activation = 'softmax')(w)
    model = Model(inputs = [base_model.input], outputs = [output])    

    
    model.summary()

    optimizer = SGD(lr = learning_rate)
    model.compile(optimizer = optimizer,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

    return model

def create_ResNet_trainable_model2(X_train, X_test, y_train, y_test, number_of_classes, learning_rate):
    
    model = Sequential()
    
    input_shape = X_train.shape[1:]    
    ResNet = ResNet50(weights='imagenet', include_top=False, input_shape = input_shape)    
    #for layer in ResNet.layers[:-20]:
    #    layer.trainable = False
        
    for layer in ResNet.layers:
        print(layer, layer.trainable)
    
    ResNet.summary()
    
    model.add(ResNet)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(number_of_classes, activation='softmax'))
    
    model.summary()

    optimizer = SGD(lr = learning_rate)
    model.compile(optimizer = optimizer,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

    return model

def create_MobileNet_trainable_model(X_train, X_test, y_train, y_test, number_of_classes, learning_rate):
    
    input_shape = X_train.shape[1:]    
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape = input_shape)  
    w = base_model.output
    w = Flatten()(w)
    #w = Dense(1024, activation='relu')(w)
    output = Dense(number_of_classes, activation = 'softmax')(w)
    model = Model(inputs = [base_model.input], outputs = [output])    
    
    #model.layers[-1].trainable = True
    #model.layers[-2].trainable = True
    #model.layers[-3].trainable = True
    #model.layers[-4].trainable = True
    #model.layers[-5].trainable = True
    
    model.summary()

    optimizer = SGD(lr = learning_rate)
    model.compile(optimizer = optimizer,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

    return model

def create_MobileNet_trainable_model2(X_train, X_test, y_train, y_test, number_of_classes, learning_rate):
    
    model = Sequential()
    
    input_shape = X_train.shape[1:]    
    mobilenet = MobileNet(weights='imagenet', include_top=False, input_shape = input_shape)  
    
    #for layer in mobilenet.layers[:-20]:
    #    layer.trainable = False
        
    for layer in mobilenet.layers:
        print(layer, layer.trainable)
        
    mobilenet.summary()
    
    model.add(mobilenet)
    model.add(Flatten())
    #model.add(Dropout(0.3))
    model.add(Dense(number_of_classes, activation='softmax'))
    
    model.summary()

    optimizer = SGD(lr = learning_rate)
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

    model.summary()

    optimizer = SGD(lr = learning_rate)
    model.compile(optimizer = optimizer,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

    return model




if __name__ == '__main__':
    print("Do nothing")