#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 14:58:01 2019
 Creating DL model
@author: artur
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

data = pd.read_csv('data',' ')

# Inputs and otuput of the model
X = data.iloc[:,1:39].values
Y = data.iloc[:,-1].values

# Data scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X,(-1,1))
#y_ = Y / Y.max(axis=0)
# Spliting data for training and testing purpose
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)





# Creating model
input_nodes = X_train.shape[1]
output_nodes =1
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers


classifier = Sequential()

classifier.add(Dense(output_dim = 38, input_dim=input_nodes, init = 'uniform', activation='relu'))  #first layer
classifier.add(Dense(output_dim = 38, init = 'uniform', activation='relu'))                         # second layer
classifier.add(Dense(output_dim = 19, init = 'uniform', activation='relu'))                         # third layer
classifier.add(Dense(output_dim = output_nodes, init = 'uniform'))            # output layer
sgd = optimizers.SGD(momentum=0.9, lr=0.0001)
classifier.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

classifier.fit(X_train,y_train,batch_size=20,epochs=1000)