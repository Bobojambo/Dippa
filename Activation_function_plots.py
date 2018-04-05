# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 13:20:02 2018

@author: hakala24
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def relu(x_values):
    title = "ReLU (x)"
    value_pairs = []
    y_values = []
    for x_value in x_values:
        if x_value > 0:
            y_value = x_value
        else:
            y_value = 0
        value_pairs.append((x_value, y_value))
        y_values.append(y_value)
    return np.array(y_values), title

def threshold(x_values):
    title = "Threshold (x)"
    value_pairs = []
    y_values = []
    for x_value in x_values:
        if x_value > 0:
            y_value = 1
        else:
            y_value = -1
        value_pairs.append((x_value, y_value))
        y_values.append(y_value)
    return np.array(y_values), title

def softmax(x):
    
    return x

def sigmoid(x_values):
    title = "Logistic sigmoid (x)"
    y_values = []
    for x_value in x_values:
        y_value = 1 / (1 + np.exp(-x_value))
        y_values.append(y_value)
    return np.array(y_values), title

def tanh(x_values):
    title = "Tanh (x)"
    y_values = []
    for x_value in x_values:
        y_value = np.tanh(x_value)
        y_values.append(y_value)
    return np.array(y_values), title

def useless():
    
        
    plt.plot([1,2,3,4])
    plt.ylabel('some numbers')
    plt.show()
    
       
    t = np.arange(0.0, 5.0, 0.01)
    s = np.cos(2*np.pi*t)
    line, = plt.plot(t, s, lw=2)
    
    plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
                arrowprops=dict(facecolor='black', shrink=0.05),
                )
    
    plt.ylim(-2,2)
    plt.show()
    
    return

if __name__ == "__main__":
     #red_patch = mpatches.Patch(color='blue', label='Sigmoid function')
    #plt.legend(handles=[red_patch])

    
    x = np.arange(-10, 10.01, 0.25)
    xticks = np.arange(x.min(), x.max()+0.01, (x.max()-x.min())/4)
    
    y, title = relu(x)
    yticks = np.arange(y.min(), y.max()+0.01, (y.max()-y.min())/4)
      
    plt.subplot(221)
    plt.plot(x, y, 'b')
    plt.title(title)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.grid(True)
    
    y, title = sigmoid(x)
    yticks = np.arange(y.min(), y.max()+0.01, (y.max()-y.min())/4)
      
    plt.subplot(222)
    plt.plot(x, y, 'b')
    plt.title(title)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.grid(True)
    
    y, title = tanh(x)
    yticks = np.arange(y.min(), y.max()+0.01, (y.max()-y.min())/4)
      
    plt.subplot(223)
    plt.plot(x, y, 'b')
    plt.title(title)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.grid(True)
    
    y, title = threshold(x)
    yticks = np.arange(y.min(), y.max()+0.01, (y.max()-y.min())/4)
      
    plt.subplot(224)
    plt.plot(x, y, 'b')
    plt.title(title)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.grid(True)
    
    plt.subplots_adjust(top=1.2, bottom=0.08, left=0.10, right=0.95, hspace=0.3,
                    wspace=0.35)
    
    plt.show()
