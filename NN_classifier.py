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
import cv2
import Image_extractor
import class_extractor
import model_builder
import generate_hyperparameters_CNN
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
from keras import metrics
import confusion_matrix_implementation
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import class_weight

class Model_class:
    
    def __init__(self, hyperparameters, history, iteration, accuracy):
        self.hyperparameters = hyperparameters
        self.history = history
        self.iteration = iteration
        self.accuracy = accuracy
        
    def print_hyperparameters(self):
        for parameter in self.hyperparameters:
            print(parameter, self.hyperparameters[parameter])
            
        #for value in the_model.hyperparameters:
        #    print(value, the_model.hyperparameters[value])
            

def load_data_from_folder(globPath = 'Images/*.jpg', test_data = False):
    
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
    
    #For testing 20 first loaded images
    
    filepaths = filepaths[:500]
    targets = targets[:500]
    
    
    return filepaths, targets




def split_data(imagedata, targets, test_size):
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(imagedata, targets, test_size = test_size, random_state = 42)
    
    return X_train, X_test, y_train, y_test

def define_target_categories(imagefilepaths, targets, banned_upper_category_labels, banned_vessel_category_labels, vessel_classification):
        
    category_labels = class_extractor.return_upper_category_target_list(targets, vessel_classification)
    
    imageData_rgb_images = []
    upper_category_labels_rgb_images = []
    i = 0
    #Check for 3 color channels, or 3 dimensions 128x128x3
    #Remove if image is not RGB image
    for imagepath in imagefilepaths:
        image = io.imread(imagepath)
        numpyImage = np.array(image)
        #io.imshow(numpyImage)
        #print(category_labels[i])
        
        #Check that the image is 3 channeled
        if numpyImage.ndim == 3:
            imageData_rgb_images.append(numpyImage)
            upper_category_labels_rgb_images.append(category_labels[i])            
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
    
    for image, label in zip(imageData_rgb_images, upper_category_labels_rgb_images):
        """For debug
        if i == 200:
            break
        i += 1
        """
        
        #if label in parent_list:
        #Check banned labels, banned labels defined in main
        
        if label == "Empty":
            continue
        
        if vessel_classification is True:
            if label in banned_vessel_category_labels:
                continue
            else:
                temp1.append(image)
                temp2.append(label)
        else:
            if label in banned_upper_category_labels:
                continue
            else:
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
    plt.title('Label distribution')
    plt.show()
    
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                    ha='center', va='bottom')
    
    return

def other_way_to_plot():
    
    # Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
     
    # Accuracy Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    
    return

def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()






def augment_data(X_train, y_train):
    
    X_train_augmented = []
    y_train_augmented = []
    for img, label in zip(X_train, y_train):
        mirror_img = img[:, ::-1]
        #mirror_img_Image = Image.fromarray(mirror_img, 'RGB')
        #mirror_img_Image.show()
        X_train_augmented.append(mirror_img)
        y_train_augmented.append(label)
    X_train_augmented = np.array(X_train_augmented)
    y_train_augmented = np.array(y_train_augmented)
    
    X_train_joined = np.concatenate((X_train, X_train_augmented))
    y_train_joined = np.concatenate((y_train, y_train_augmented))
    #X_train = np.append(X_train, X_train_augmented)
    
    return X_train_joined, y_train_joined


def load_data(vessel_classification, data_augmentation, image_size = 96, call_from_CNN_generator = False):
    

    banned_upper_category_labels = ["Ice", "Weather phenomen", "Misc object", "Island"]
    banned_vessel_category_labels = []
    #vessel_subclasses = ["Cargo vessel", "Fishing vessel", "High speed craft", "Icebreaker", "Offshore vessel", "Passenger vessel", "Pleasure craft", "Military vessel", "Special vessel", "Tug"]
    
    if image_size == 32:
        globPath = 'ResizedImages32x32/*.jpg'
    elif image_size == 64:
        globPath = 'ResizedImages64x64/*.jpg'
    elif image_size == 96:
        globPath = 'ResizedImages96x96/*.jpg'
    elif image_size == 128:
        globPath = 'ResizedImages128x128/*.jpg'
    elif image_size == 224:
        globPath = 'ResizedImages224x224/*.jpg'
        
    parent_list, dictionaries = class_extractor.return_class_dictionaries()    
    imagefilepaths, targets = load_data_from_folder(globPath)
    finalData, labels = define_target_categories(imagefilepaths, targets, banned_upper_category_labels, banned_vessel_category_labels, vessel_classification)
    data_loaded = True            
    finalData = np.array(finalData)        

    counter = Counter(labels)
    number_of_classes = len(counter)
    
    lb = LabelBinarizer()
    lb.fit(labels)
    labels_binarized = lb.transform(labels)

    #model implementation doesnt work without "if class amount == 2"
    if number_of_classes == 2:
        labels_binarized = to_categorical(labels_binarized, num_classes=2)    
    
    
    #Split data (CrossValidation needed)
    test_size = 0.2
    X_train, X_test, y_train, y_test = split_data(finalData, labels_binarized, test_size)
    
    if data_augmentation == True:
        print("data augmentation")
        X_train, y_train = augment_data(X_train, y_train)
        
    return X_train, X_test, y_train, y_test, labels, number_of_classes, lb, data_loaded

def plot_history_LOSS_ACCURACY_DOTPLOT(history, model, learning_rate):
    
    learning_rate_string = "%.7f" % learning_rate
    title = "learning rate: " + learning_rate_string
    
    fig, ax = plt.subplots(2, 1)
    ax[0].set_title(title)
    ax[0].plot(history.history['acc'], 'ro-', label = "Train Accuracy")
    ax[0].plot(history.history['val_acc'], 'go-', label = "Test Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend(loc = "best")
    ax[0].grid('on')
    
    ax[1].plot(history.history['loss'], 'ro-', label = "Train Loss")
    ax[1].plot(history.history['val_loss'], 'go-', label = "Test Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend(loc = "best")
    ax[1].grid('on')
    
    plt.tight_layout()
    plt.show()
    try:
        plt.savefig("Last_results/Accuracy.pdf", bbox_inches = "tight")
    except:
        print("error in figure saving")
        return history, model
    
    return

def train_model(model, X_train, X_test, y_train, y_test, learning_rate, labelbinarizer, n_epochs = 1, batch_size = 32):
    
    #Class weight implementation
    lb = labelbinarizer
    training_labels = lb.inverse_transform(y_train)
    le = LabelEncoder()
    le.fit(training_labels)
    training_labels_integers = le.transform(training_labels)
    class_weights_for_training = class_weight.compute_class_weight('balanced', np.unique(training_labels_integers), training_labels_integers)
    #Input the dict to model.fit(class_weight)
    class_weights_dict_for_training = dict(enumerate(class_weights_for_training))
    
    history = model.fit(X_train, y_train, epochs = n_epochs, batch_size = batch_size, class_weight = class_weights_dict_for_training,
            validation_data = (X_test, y_test))

    plot_history_LOSS_ACCURACY_DOTPLOT(history, model, learning_rate)
    
    return history, model

if __name__ == '__main__':
    
    if 'data_loaded' not in locals():
        
        input_var = input("Vessel Classification (True/False): ")
        if input_var == "True":
            vessel_classification = True
        else:
            vessel_classification = False
        
        input_var = input("Use Data Augmentation (True/False): ")
        if input_var == "True":
            data_augmentation = True
        else:
            data_augmentation = False
            
        X_train, X_test, y_train, y_test, labels, number_of_classes, lb, data_loaded \
        = load_data(vessel_classification, data_augmentation)
    
    else:
        print("Data already loaded")

    
    input_var = input("Generate CNN models (True/False): ")
    
    if input_var == "True":            
        model_list = []
        number_of_models = 2
        n_epochs = 50
        batch_size = 32
        best_model_KERAS_MODEL = None
        best_model_MODEL_CLASS = None
        best_val_accuracy = 0
        model_index = 0
        # i number of random generated CNN models
        i = 0           
        while i < number_of_models:
            
            hyperparameters = generate_hyperparameters_CNN.generate_hyperparameters()
            input_size = hyperparameters.get("input size")
            X_train, X_test, y_train, y_test, labels, number_of_classes, lb, data_loaded = load_data(vessel_classification, data_augmentation, input_size)
            
            
            model = generate_hyperparameters_CNN.generate_CNN(X_train, X_test, y_train, y_test, number_of_classes, hyperparameters)
            learning_rate = hyperparameters.get("learning rate")           
            history, model = train_model(model, X_train, X_test, y_train, y_test, learning_rate, lb, n_epochs, batch_size) 
            
            #Plotting, figures etc
            plot_history(history)
            confusion_matrix_implementation.prepare_confusion_matrix(lb, model, labels, X_test, y_test)
            
            #Saving Model class, checking for best model. If best model, save model
            #for later processing and save model class for parameter checking
            loss_and_accuracy = model.evaluate(X_test, y_test)            
            the_model = Model_class(hyperparameters, history, i, loss_and_accuracy[1])
            
            if the_model.accuracy > best_val_accuracy:
                best_model_KERAS_MODEL = model
                best_model_MODEL_CLASS = the_model                
            model_list.append(the_model)
            
            i += 1
        
        #Show the results and hyperparameters of the best model
        plot_history(history)
        confusion_matrix_implementation.prepare_confusion_matrix(lb, model, labels, X_test, y_test)            
        best_model_MODEL_CLASS.print_hyperparameters()
        
        #Old method for finding the best model
        """
        for model_class in model_list:            
            if model_class.accuracy > best_val_accuracy:
                best_val_accuracy = model_class.accuracy
                model_index = model_class.iteration
        """


    input_var = input("Train VGG16 (True/False): ")
    if input_var == "True":
        #Training information
        n_epochs = 10
        batch_size = 32
        #learning_rates = [10**-7 , 10**-6, 10**-5, 10**-4 , 10**-3, 10**-2, 10**-1]
        learning_rate = 10**-4
        
        model = model_builder.create_VGG16_trainable_model(X_train, X_test, y_train, y_test, number_of_classes, learning_rate)
        history, model = train_model(model,  X_train, X_test, y_train, y_test, learning_rate, lb, n_epochs, batch_size)
        plot_history(history)
        confusion_matrix_implementation.prepare_confusion_matrix(lb, model, labels, X_test, y_test)
        
    input_var = input("Train ResNet (True/False): ")
    if input_var == "True":
        #Training information
        n_epochs = 10
        batch_size = 32
        learning_rate = 10**-4
        
        model = model_builder.create_ResNet_trainable_model(X_train, X_test, y_train, y_test, number_of_classes, learning_rate)
        history, model = train_model(model,  X_train, X_test, y_train, y_test, learning_rate, lb, n_epochs, batch_size)
        plot_history(history)
        confusion_matrix_implementation.prepare_confusion_matrix(lb, model, labels, X_test, y_test)
        
    input_var = input("Train Mobilenet (True/False): ")
    if input_var == "True":
        #Training information
        n_epochs = 10
        batch_size = 32
        learning_rate = 10**-4
        
        model = model_builder.create_MobileNet_trainable_model(X_train, X_test, y_train, y_test, number_of_classes, learning_rate)
        history, model = train_model(model, X_train, X_test, y_train, y_test, learning_rate, lb, n_epochs, batch_size)
        plot_history(history)
        confusion_matrix_implementation.prepare_confusion_matrix(lb, model, labels, X_test, y_test)
    
    """


    
    history_list = []
    
    for learning_rate in learning_rates:
        #model = model_builder.create_VGG16_trainable_model(X_train, X_test, y_train, y_test, number_of_classes, learning_rate)
        
        #model = model_builder.create_VGG16_freezed_model(X_train, X_test, y_train, y_test, number_of_classes, learning_rate)

        history, model = train_model(model,  X_train, X_test, y_train, y_test, learning_rate, n_epochs, batch_size) 
        plot_history(history)
        history_list.append(history)
        confusion_matrix_implementation.prepare_confusion_matrix(lb, model, labels, X_test, y_test)
        
    #Label chart
    #plot_labels_barchart(labels)
    """

















"""
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

"""

    
