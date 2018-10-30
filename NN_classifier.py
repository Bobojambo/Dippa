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
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
import confusion_matrix_implementation
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight
import matplotlib.patches as mpatches
import time
import timeit
from matplotlib import colors as mcolors
import json
import pandas
#from pandas_ml import ConfusionMatrix

class Model_class:
    
    def __init__(self, hyperparameters = None, history = None, iteration = None, accuracy = None, lb = None, learning_rate = None, y_pred = None):
        self.hyperparameters = hyperparameters
        self.history = history
        self.iteration = iteration
        self.accuracy = accuracy
        self.labelBinarizer = lb
        self.learning_rate = learning_rate
        self.y_pred = y_pred
        
    def print_hyperparameters(self):
        for parameter in self.hyperparameters:
            print(parameter, self.hyperparameters[parameter])
    
    def set_name(self, name):
        self.name = name


def save_history():
    history_dict = model_list[0].history
    
    return

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
    
    #For testing 500 first loaded images  
    if test_data == True:
        filepaths = filepaths[:1000]
        targets = targets[:1000]    
        
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
    
    i = 0
    temp1 = []
    temp2 = []    
    #Remove categories not in parent list
    #Current filter in class extractor; these categories are not in any subgroup
    
    for image, label in zip(imageData_rgb_images, upper_category_labels_rgb_images):        
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
            if label in banned_vessel_category_labels:
                continue
            else:
                temp1.append(image)
                temp2.append(label)

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

    # this is for plotting purpose
    index = np.arange(len(label_names))
    plt.bar(index, label_amounts)
    plt.xlabel('Label', fontsize=10)
    plt.ylabel('Number of images', fontsize=10)
    plt.xticks(index, label_names, fontsize=10, rotation=30)
    plt.title('Label distribution')
    plt.show()
    
    return

def annot_max(x,y, ax=None):
    
    xmax = x[np.argmax(y)]
    ymax = max(y)
    #text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    text = "esa"
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)    
    return


def plot_history(history, learning_rate=None):
    
    #for history in historys:
    
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
    
    
    
    #plt.title('Loss')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.savefig("Last_results/CNN_loss.pdf", bbox_inches = "tight")
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    

    #plt.title('Accuracy')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.savefig("Last_results/CNN_accuracy.pdf", bbox_inches = "tight")
    
    return
"""
def history_pring_testing(history):
    
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    ## As loss always exists
    epochs = range(1,len(history.history[acc_list[0]]) + 1)
    
    fig = plt.figure()
    #ax = fig.add_subplot(111)
    
    annot_max(epochs, acc_list)
    
    print("max accuracy: ", max(val_acc_list))
    
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    
    #ax.annotate('local max',xy=(10, 0.7), xytext=(11, 0.8), arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.show()


    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    
    plt.show()
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.show()    

    return
"""
def plot_multiple_histories(model_list):
    
    #epochs = 50
    model_names = ["Model1", "Model2", "Model3", "Model4", "Model5"]
    colors = ['r', 'c', 'm', 'k', '#808000']
    colors_loss =  ['r--', 'c--', 'm--', 'k--', '#808000--']
    i = 0
    #for history in historys:
    for model in model_list:
        history = model.history
        
        loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
        #acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
        val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
        
        ## As loss always exists
        epochs = range(1,len(history.history[val_acc_list[0]]) + 1)   
        
        ## Accuracy
        plt.figure(1)
        #for l in val_acc_list:    
        #    plt.plot(epochs, history.history[l], colors[i], label= str(model_names[i]) + ' (' + str(format(history.history[l][-1],'.5f'))+')')
        for l in loss_list:    
            plt.plot(epochs, history.history[l], colors[i], label= str(model_names[i]) + ' (' + str(format(history.history[l][-1],'.5f'))+')')
        for l in val_loss_list:
            plt.plot(epochs, history.history[l], colors_loss[i], label= str(model_names[i]) + ' ('  + str(str(format(history.history[l][-1],'.5f'))+')'))
        i += 1
        #annot_max(epochs, history.history['acc'])

    #plt.title('Accuracy')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.legend(loc=(1.04,0.5))
    plt.savefig("Last_results/Val_accuracies.pdf", bbox_inches = "tight")
    
    plt.show()    
    
    return

def augment_data(X_train, y_train):
    
    X_train_augmented = []
    y_train_augmented = []
    for img, label in zip(X_train, y_train):
        mirror_img = img[:, ::-1]
        #mirror_img_Image = Image.fromarray(mirror_img, 'RGB')
        #mirror_img_Image.show()
        X_train_augmented.append(mirror_img)
        y_train_augmented.append(label)
        
    X_train = np.append(X_train, np.array(X_train_augmented),axis = 0)
    y_train = np.append(y_train, np.array(y_train_augmented),axis = 0)
    
    #X_train_joined = np.concatenate((X_train, X_train_augmented))
    #y_train_joined = np.concatenate((y_train, y_train_augmented))
    
    return X_train, y_train


def load_data(vessel_classification, data_augmentation, image_size = 224, call_from_CNN_generator = False, test_data = False):
    
    banned_upper_category_labels = ["Ice", "Weather phenomen", "Misc object", "Island"]
    #banned_upper_category_labels = []
    banned_vessel_category_labels = ["Icebreaker", "Military vessel"]
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
    imagefilepaths, targets = load_data_from_folder(globPath, test_data)
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
    finalData = None #Empty for memory
    if data_augmentation == True:
        print("data augmentation")
        X_train, y_train = augment_data(X_train, y_train)
        
    return X_train, X_test, y_train, y_test, labels, number_of_classes, lb, data_loaded


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
    
    #early stopping
    #ORIGINAL PATIENCE = 10
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')  
    
    #Learning rate update
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1,
                                  patience=5, min_lr=0.00001)       
    #learning_rate_changer = LearningRateScheduler(reduce_lr)
    
    #CSV saving
    csv_logger = CSVLogger('training.csv', append=True, separator=',')
    
    # checkpoint
    best_weights_filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(best_weights_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    
    callbacks_list = [checkpoint, earlyStopping, reduce_lr, csv_logger]
    #callbacks_list = [csv_logger]
    
    history = model.fit(X_train, y_train, epochs = n_epochs, batch_size = batch_size, callbacks = callbacks_list, class_weight = class_weights_dict_for_training,
            validation_data = (X_test, y_test))
    #model.load_weights(best_weights_filepath)
    #plot_history_LOSS_ACCURACY_DOTPLOT(history, model, learning_rate)
    
    
    return history, model

def predict_y_pred_for_X_test(lb, model, labels, X_test, y_test):
    
        #--------------Own implementation STARTS---------------------------
    #Confusion matrix
    le = LabelEncoder()
    le.fit(labels)        
    y_test_labels = lb.inverse_transform(y_test)
    y_test_labels_encoded = le.transform(y_test_labels)
    
    numpy_list_labels = np.array(labels)
    class_names = np.unique(numpy_list_labels)
    
    #Continuous values
    y_pred = model.predict(X_test)
    #LabelEncoded y_pred
    y_pred = y_pred.argmax(axis=1)
    
    return y_pred
    #--------------Own implementation ENDS---------------------------
    
def shore_structure_missclassification():
    
    y_test_labels = lb.inverse_transform(y_test)
    structures = []
    i = 0
    for test_label in y_test_labels:
        if test_label == "Structure":
            structures.append(X_test[i])     
        i += 1
    #structures = [X_test[16],X_test[32],X_test[41],X_test[101],X_test[106],X_test[120], X_test[132],X_test[137],X_test[143],X_test[169],X_test[176]]
    structures = np.array(structures)
    structures_predictions = model.predict(structures)
    structures_predictions_classes = structures_predictions.argmax(axis=1)
    structures_predictions_classes_binarized = lb.transform(structures_predictions_classes)
    #structures_predictions = lb.transform(structures_predictions)
    
    img = Image.fromarray(structures[46], 'RGB')
    img.save('structure_correct1.jpg')
    img = Image.fromarray(structures[70], 'RGB')
    img.save('structure_correct2.jpg')
    img = Image.fromarray(structures[130], 'RGB')
    img.save('structure_correct3.jpg')
    img = Image.fromarray(structures[293], 'RGB')
    img.save('structure_incorrect1.jpg')
    img = Image.fromarray(structures[197], 'RGB')
    img.save('structure_incorrect2.jpg')
    img = Image.fromarray(structures[355], 'RGB')
    img.save('structure_incorrect3.jpg')
    
    img = Image.fromarray(structures[10], 'RGB')
    img.show()
    
    return

def save_test_images_to_folder():
    
    path = "TestImages/"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)  
    
    y_test_labels = lb.inverse_transform(y_test)
    i = 0
    for image, label in zip(X_test, y_test_labels):
        img = Image.fromarray(image)
        img.save("{}image{}{}.jpg".format(path,i, label))
        i += 1
        
        
    path = "TestImages/Structures/"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)  
    
    y_test_labels = lb.inverse_transform(y_test)
    i = 0
    for image, label in zip(X_test, y_test_labels):
        if label == "Structure":
            img = Image.fromarray(image)
            img.save("{}image{}{}.jpg".format(path,i, label))
            i += 1
    
    return
    
def time_for_classification(model, X_test):   
    """
    start = time.time()
    model.predict(X_test)
    end = time.time()
    print("Total time predicting X_test:", end - start)
    print("Averaged X_test: ", (end - start)/len(X_test))
    start = time.time()
    model.predict(X_test[:1])
    end = time.time()
    print("Single image time", end - start)
    """
    i=0
    measured_times = []
    kuvat_kasitelty = []
    for kuva in X_test:
        image = X_test[i][np.newaxis, ...]
        kuvat_kasitelty.append(image)
        start = time.time()
        model.predict(image)
        end = time.time()
        measured_times.append(end-start)
        i = i + 1
        if i == 100:
            break
    summa = 0
    for value in measured_times:
        summa += value
    keskiarvo = summa/len(measured_times)
    print(keskiarvo)
    keskiarvo = np.mean(measured_times)
    varianssi = np.var(measured_times)
    print("keskiarvo: ", keskiarvo, " varianssi: ", varianssi)
    time_for_classification2(model, kuvat_kasitelty)
    return #measured_times

def time_for_classification2(model, images):
    measured_times = []
    for image in images:
        start = time.time()
        model.predict(image)
        end = time.time()
        measured_times.append(end-start)
    summa = 0
    for value in measured_times:
        summa += value
    keskiarvo = summa/len(measured_times)
    print(keskiarvo)
    keskiarvo = np.mean(measured_times)
    varianssi = np.var(measured_times)
    print("keskiarvo2: ", keskiarvo, " varianssi2: ", varianssi)
    return

if __name__ == '__main__':
    
    input_var = input("Load Data (True/False): ")
    if input_var == "True":
        
        input_var = input("Test Data Size (True/False): ")
        if input_var == "True":
            test_data = True
        else:
            test_data = False
        
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
        = load_data(vessel_classification, data_augmentation, test_data = test_data)
    
    else:
        print("Data already loaded")

    if 'history_list' not in locals():
        history_list = []

    if 'model_list' not in locals():
        model_list = []
    
    input_var = input("Generate CNN models (True/False): ")
    
    if input_var == "True":
        number_of_models = 1
        n_epochs = 1
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
            X_train, X_test, y_train, y_test, labels, number_of_classes, lb, data_loaded = \
                load_data(vessel_classification, data_augmentation, image_size = input_size, test_data = test_data)
            
            
            model = generate_hyperparameters_CNN.generate_CNN(X_train, X_test, y_train, y_test, number_of_classes, hyperparameters)
            learning_rate = hyperparameters.get("learning rate")           
            history, model = train_model(model, X_train, X_test, y_train, y_test, learning_rate, lb, n_epochs, batch_size) 
            
            #Plotting, figures etc
            plot_history(history)            
            confusion_matrix_implementation.prepare_confusion_matrix(lb, model, labels, X_test, y_test)
            
            #Saving Model class, checking for best model. If best model, save model
            #for later processing and save model class for parameter checking
            loss_and_accuracy = model.evaluate(X_test, y_test)
            
            the_model = Model_class(hyperparameters, history, i, loss_and_accuracy[1], lb)
            
            if the_model.accuracy > best_val_accuracy:
                best_val_accuracy = the_model.accuracy
                best_model_KERAS_MODEL = model
                best_model_MODEL_CLASS = the_model               
                
            model_list.append(the_model)
            
            i += 1    
        
        #Show the results and hyperparameters of the best model
        plot_history(best_model_MODEL_CLASS.history)
        
        #other_way_to_plot(best_model_MODEL_CLASS.history)
        
        #Load data for conf-matrix implementation        
        best_model_input_size = best_model_MODEL_CLASS.hyperparameters.get("input size")        
        X_train, X_test, y_train, y_test, labels, number_of_classes, lb, data_loaded = \
                load_data(vessel_classification, data_augmentation, image_size = best_model_input_size, test_data = test_data)
        
        confusion_matrix_implementation.prepare_confusion_matrix(best_model_MODEL_CLASS.labelBinarizer, best_model_KERAS_MODEL, labels, X_test, y_test)            
        
        best_model_MODEL_CLASS.print_hyperparameters()
        
        time_for_classification(model, X_test)
        
        #Old method for finding the best model
        """
        for model_class in model_list:            
            if model_class.accuracy > best_val_accuracy:
                best_val_accuracy = model_class.accuracy
                model_index = model_class.iteration
        """
    input_var = input("Train BigModels (True/False): ")
    if input_var == "True":

        #Training info for VGG, ResNet and MobileNet---------------------------------------------------------
        hyperparameters = []
        i = 0
        n_epochs = 1
        batch_size = 32
        learning_rate = 10**-4
        hyperparameters = 0
        best_val_accuracy = 0
        model_trained = False
    
        input_var = input("Train VGG16 (True/False): ")
        if input_var == "True":            
                
            model = model_builder.create_VGG16_trainable_model2(X_train, X_test, y_train, y_test, number_of_classes, learning_rate)            
            history, model = train_model(model,  X_train, X_test, y_train, y_test, learning_rate, lb, n_epochs, batch_size)
            model_trained = True
                    
        input_var = input("Train ResNet (True/False): ")
        if input_var == "True":
            
            model = model_builder.create_ResNet_trainable_model2(X_train, X_test, y_train, y_test, number_of_classes, learning_rate)
            history, model = train_model(model,  X_train, X_test, y_train, y_test, learning_rate, lb, n_epochs, batch_size)   
            model_trained = True
            
            for layer in model.layers:
                print(layer, layer.trainable)
            
        input_var = input("Train Mobilenet (True/False): ")
        if input_var == "True":

            #model = model_builder.create_MobileNet_trainable_model(X_train, X_test, y_train, y_test, number_of_classes, learning_rate)
            model = model_builder.create_MobileNet_trainable_model2(X_train, X_test, y_train, y_test, number_of_classes, learning_rate)        
            history, model = train_model(model, X_train, X_test, y_train, y_test, learning_rate, lb, n_epochs, batch_size)
            model_trained = True
                    
        #After training session------------------------------------------------------------------------------
            
        loss_and_accuracy = model.evaluate(X_test, y_test)      
        y_pred = predict_y_pred_for_X_test(lb, model, labels, X_test, y_test)
        the_model = Model_class(hyperparameters, history, i, loss_and_accuracy[1], lb, learning_rate, y_pred)  
        
        if the_model.accuracy > best_val_accuracy:
            best_val_accuracy = the_model.accuracy
            best_model_KERAS_MODEL = model
            best_model_MODEL_CLASS = the_model        
        
        model_list.append(the_model) 
        #Plot histories
        plot_history(best_model_MODEL_CLASS.history, learning_rate)
        #Confusion matrix
        confusion_matrix_implementation.prepare_confusion_matrix(best_model_MODEL_CLASS.labelBinarizer, best_model_KERAS_MODEL, labels, X_test, y_test)  
        #Print hyperparameters
        #best_model_MODEL_CLASS.print_hyperparameters()
        #Plot labels    
        plot_labels_barchart(labels)
        #Time for classification
        time_for_classification(model, X_test)
        #Plot multiple history ins ame figure
        plot_multiple_histories(model_list)



        #model.save_weights("lastmodel")
        #model.load_weights(filepath, by_name=False)