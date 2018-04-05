# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 12:42:35 2018

@author: TaitavaBobo§
"""

print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

def load_testdata():

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names
    
    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    # Run classifier, using a model that is too regularized (C too low) to see
    # the impact on the results
    classifier = svm.SVC(kernel='linear', C=0.01)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    
    return


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

def prepare_confusion_matrix(lb, model, labels, X_test, y_test):
    
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
    #--------------Own implementation ENDS---------------------------
    
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test_labels_encoded, y_pred)
    np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')        
    plt.tight_layout()
    plt.savefig("Last_results/Matrix.pdf", bbox_inches = "tight")
    
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.tight_layout()
    plt.savefig("Last_results/MatrixNormalized.pdf", bbox_inches = "tight")
    
    plt.show()
    
    return

if __name__ == '__main__':
    print("nothing")