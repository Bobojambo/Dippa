# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:41:41 2018

@author: TaitavaBobo§
"""

import numpy as np
from keras.models import Sequential
from sklearn import preprocessing
from sklearn import model_selection

from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD



def read_data():

    #x_test = np.load("X_test.npy")
    #data = np.load("X_train.npy")
    data = []
    targets = []
    with open('Classes/classes.csv', 'r') as fp:
        for line in fp:
            # Otherwise, split the line
            values = line.split(",")
            targets.append(values[1].strip())
    fp.close()
    
    targets = np.array(targets)
        
    return data, targets

def create_VGG_CNN_model():
    
    num_classes = 15
    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(40, 501, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])
    
    
    return model

def create_CNN_model():
    
    num_featmaps = 32 # This many filters per layer
    num_classes = 15 # Digits 0,1,...,9
    #num_epochs = 50 # Show all samples 50 times
    w, h = 5, 5 # Conv window size
        
    model = Sequential()
    
    # Layer 1: needs input_shape as well.
    model.add(Conv2D(num_featmaps, (w, h),
    input_shape=(40, 501, 1),
    activation = 'relu'))
    
    # Layer 2:
    model.add(Conv2D(num_featmaps, (w, h), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Layer 3: dense layer with 128 nodes
    # Flatten() vectorizes the data:
    # 32x10x10 -> 3200
    # (10x10 instead of 14x14 due to border effect)
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    
    # Layer 4: Last layer producing 10 outputs.
    model.add(Dense(num_classes, activation='softmax'))
    # Compile and train
    model.compile(loss='categorical_crossentropy',
    optimizer='adadelta',
    metrics = ['accuracy'])
    return model
    

def main():
    
    data, targets = read_data()
    print(targets)
    
    #Smaller data for CPU compute testing
    
    #data = data[:100]
    #targets = targets [:100]
        
    #Label encoder changing string values to integers for labeling
    #le = preprocessing.LabelEncoder()
    #le.fit(targets)
    #targets = le.transform(targets)
    
    
    #targets = to_categorical(targets)

def write_submission(model, x_test, le):

    
    # Assume that your LabelEncoder is called le.
    y_pred = model.predict(x_test)
    preds = []
    for row in y_pred:
        preds.append(np.argmax(row))
    
    labels = list(le.inverse_transform(preds))
    with open("submission.csv", "w") as fp:
        fp.write("Id,Scene_label\n")
        for i, label in enumerate(labels):
            fp.write("%d,%s\n" % (i, label))    

if __name__ == '__main__':
    main() # this calls main function