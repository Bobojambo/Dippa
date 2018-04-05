# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:41:41 2018

@author: TaitavaBobo§
"""

from collections import Counter
from random import randint
from skimage import data, io, filters
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import shutil
import sys
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
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.applications.vgg16 import VGG16
import confusion_matrix_implementation
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


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

def create_CNN_model(X_train, X_test, y_train, y_test, number_of_classes):
    

    #l1 = Regularizer()
    input_shape = X_train.shape[1:]

    model = Sequential()

    model.add(Conv2D(filters = 96, 
                    strides=4,
                    kernel_size = (11, 11),
                    activation = 'relu',
                    input_shape=input_shape))
   #l1 = Regularizer()

    
    model = Sequential()
    
    model.add(Conv2D(filters = 32, 
                     kernel_size = (5, 5),
                     activation = 'relu',
                     padding = 'same',
                     input_shape=(128,128,3)))    
    model.add(MaxPooling2D(2,2))

   
    model.add(MaxPooling2D(2,2))
   #model.add(Dropout(0.25))
   
    model.add(Conv2D(filters = 256, 
    kernel_size = (5, 5),
    padding='same',
    activation = 'relu'))
       #model.add(Dropout(0.2))   
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(filters = 384, 
                 kernel_size = (3, 3),
                 padding='same',
                 activation = 'relu'))
   #model.add(Dropout(0.1))
    model.add(Conv2D(filters = 384, 
                 kernel_size = (3, 3),
                 padding='same',
                 activation = 'relu'))
                  
    model.add(Conv2D(filters = 384, 
                 kernel_size = (3, 3),
                 padding='same',
                 activation = 'relu'))
                   
    model.add(Conv2D(filters = 256, 
                 kernel_size = (3, 3),
                 padding='same',
                 activation = 'relu'))
    

    model.add(Flatten())
    model.add(Dense(2048, activation = 'relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(256, activation = 'relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(number_of_classes, activation='softmax'))

    model.summary()

   #optimizer = SGD(lr = 1e-0)
    optimizer = SGD(lr = 0.0001)
    model.compile(optimizer = optimizer,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

    return model

def VGG16_model(X_train, X_test, y_train, y_test, number_of_classes, learning_rate):
    
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

def create_CNN_model_small(X_train, X_test, y_train, y_test, number_of_classes, learning_rate):
    
        #l1 = Regularizer()
    input_shape = X_train.shape[1:]

    model = Sequential()

    model.add(Conv2D(filters = 32,
                    kernel_size = (5, 5),
                    padding = 'same',
                    activation = 'relu',
                    input_shape=input_shape))
    
    model.add(Conv2D(filters = 64, 
                     kernel_size = (5, 5),
                     activation = 'relu',
                     padding = 'same'))    
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(filters = 64, 
                    kernel_size = (5, 5),
                    padding='same',
                    activation = 'relu'))

    model.add(Conv2D(filters = 64, 
                 kernel_size = (3, 3),
                 padding='same',
                 activation = 'relu'))
    model.add(MaxPooling2D(2,2))
   #model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(256, activation = 'relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(number_of_classes, activation='softmax'))

    model.summary()

   #optimizer = SGD(lr = 1e-0)
    optimizer = SGD(lr = learning_rate)
    model.compile(optimizer = optimizer,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])
    
    return model


def train_model(model, X_train, X_test, y_train, y_test, learning_rate, n_epochs = 1, batch_size = 32):

    history = model.fit(X_train, y_train, epochs = n_epochs, batch_size = batch_size,
            validation_data = (X_test, y_test))

    learning_rate_string = "%.7f" % learning_rate
    title = "learning rate: " + learning_rate_string
    
    fig, ax = plt.subplots(2, 1)
    ax[0].set_title(title)
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
    plt.savefig("Last_results/Accuracy.pdf", bbox_inches = "tight")
    
    return history, model


def split_data(imagedata, targets, test_size):
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(imagedata, targets, test_size = test_size)
    
    return X_train, X_test, y_train, y_test

def define_target_categories(imagefilepaths, targets, depth = 1):
        
    upper_category_labels, vessel_child_target_labels = class_extractor.return_upper_category_target_list(targets)
    #counter = Counter(vessel_child_target_labels)
    #print(counter)
    #Calculate the amount of labels per category
    #categories_upper_labels = Counter(upper_category_labels)
    
    imageData_rgb_images = []
    upper_category_labels_rgb_images = []
    i = 0
    #Check for 3 color channels, or 3 dimensions 128x128x3
    #Remove if image is not RGB image
    for imagepath in imagefilepaths:
        image = io.imread(imagepath)
        numpyImage = np.array(image)
        #io.imshow(numpyImage)
        #print(upper_category_labels[i])
        
        #Check that the image is 3 channeled
        if numpyImage.ndim == 3:
            imageData_rgb_images.append(numpyImage)
            upper_category_labels_rgb_images.append(upper_category_labels[i])            
        i = i + 1

    """ For debug purposes save used images to folder
    path = "ImagesUsed/"        
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)    
    
    index = 0
    for image, label in zip(imageData_rgb_images, upper_category_labels_rgb_images):
        #image.show()
        filename = "{}image{}{}.jpg".format(path,index,label)
        io.imsave(filename, image)
        index += 1
        if index == 200:
            break
    """
    
    i = 0
    temp1 = []
    temp2 = []    
    #Remove categories not in parent list
    #Current filter in class extractor; these categories are not in any subgroup
    label_counter = {}
    
    for image, label in zip(imageData_rgb_images, upper_category_labels_rgb_images):
        """For debug
        if i == 200:
            break
        i += 1
        """
        
        if label in parent_list:
            #Remove ice and weather phenomen because not enough data
            if label == "Ice" or label == "Weather phenomen" or label == "Misc object" or \
            label == "Island" or label == "Shore" or label == "Vessel" or label == "Living being" or label == "Structure":
                continue
            else:
                #if label not in label_counter:
                #    label_counter[label] = 1
                #else:
                #    if label_counter.get(label) > 500:
                #        i = i + 1
                #        continue
                #    label_counter[label] += 1
                    
                temp1.append(image)
                temp2.append(label)
                
        #else:
            #print("label not in parent list")
            #sys.exit()
    
    """    For debug purposes save images to folder
    path = "ImagesUsedFiltered/"        
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    
    
    index = 0
    for image, label in zip(temp1, temp2):
        #image.show()
        filename = "{}image{}{}.jpg".format(path,index,label)
        io.imsave(filename, image)
        index += 1         
    #"""
    
    finalData = temp1
    categories = temp2
    
    return finalData, categories

def plot_labels_barchart(labels):

    LabelCounter = Counter(labels)
    label_names = []
    label_amounts = []
    for label in LabelCounter:
        amount = LabelCounter.get(label)
        label_names.append(label)
        label_amounts.append(amount)
    #Amount of different labels
    #number_of_classes = len(counter)

    # this is for plotting purpose
    index = np.arange(len(label_names))
    plt.bar(index, label_amounts)
    plt.xlabel('Label', fontsize=10)
    plt.ylabel('Number of images', fontsize=10)
    plt.xticks(index, label_names, fontsize=10, rotation=30)
    plt.title('Data distribution over labels')
    plt.show()
    
    return

if __name__ == '__main__':
    
    if 'data_loaded' not in locals():    
        parent_list, dictionaries = class_extractor.return_class_dictionaries()    
        imagefilepaths, targets = load_data(globPath = 'ResizedImages/*.jpg')
        #imagefilepaths, targets = load_data(globPath = 'ResizedImages64x64/*.jpg')
        finalData, labels = define_target_categories(imagefilepaths, targets, depth = 1)
        #X_train, X_test, y_train, y_test = split_data(finalData, labels) 
        data_loaded = True    
        
        finalData = np.array(finalData)
        
        counter = Counter(labels)
        number_of_classes = len(counter)
    
    #For test purposes
    if len(labels) > 10:       

        
        lb = LabelBinarizer()
        lb.fit(labels)
        labels_binarized = lb.transform(labels)
        #model implementation doesnt work if class amount == 2
        if number_of_classes == 2:
            labels_binarized = to_categorical(labels_binarized, num_classes=2)
        
        #Split data (CrossValidation needed)
        test_size = 0.2
        X_train, X_test, y_train, y_test = split_data(finalData, labels_binarized, test_size)
        
        
        #Define hyperparameters
        #learning_rates = [10**-7 , 10**-6, 10**-5, 10**-4 , 10**-3, 10**-2, 10**-1]
        learning_rates = [10**-5]
                    
        #Training information
        n_epochs = 75
        batch_size = 32
        
        results = []
        
        for learning_rate in learning_rates:
            #Create model
            #model = create_CNN_model(X_train, X_test, y_train, y_test, number_of_classes)        
            model = VGG16_model(X_train, X_test, y_train, y_test, number_of_classes, learning_rate)
            #model = create_CNN_model_small(X_train, X_test, y_train, y_test, number_of_classes, learning_rate)
            

            history, model = train_model(model,  X_train, X_test, y_train, y_test, learning_rate, n_epochs, batch_size)            
    
            confusion_matrix_implementation.prepare_confusion_matrix(lb, model, labels, X_test, y_test)
            
        #Label chart
        plot_labels_barchart(labels)

def create_basic_classifiers():
    models = []    
    #Randomforst, extratrees, gradboost, adaboost
    esa = False
    if esa == True:
        X_train = np.ravel(X_train)
        X_test = np.ravel(X_test)
        n_trees = 100
        clf_list2 = [RandomForestClassifier(n_estimators = n_trees), ExtraTreesClassifier(n_estimators = n_trees), GradientBoostingClassifier(n_estimators = n_trees), AdaBoostClassifier(n_estimators = n_trees) ]
        clf_name2 = ["Random Forest", "Extra Trees", "GradientBoost", "Adaboost"]
        for clf,name in zip(clf_list2, clf_name2):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            print(name, " Score: ", score)   
    
    return models



    
