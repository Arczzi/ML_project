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
X = data.iloc[:,0:40].values
Y = data.iloc[:,-1].values

# Point threshold
th = 1
# Data scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
#y_ = Y / Y.max(axis=0)
# Spliting data for training and testing purpose
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,Y,test_size=0.01)

# Creating model
input_nodes = X_train.shape[1]
output_nodes =1



from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

classifier = Sequential()
classifier.add(Dense(output_dim = input_nodes, input_dim=input_nodes, activation='relu'))   #first layer (input) -> second 38 nodes
classifier.add(Dense(output_dim = input_nodes, activation='relu'))                          # second layer -> third layer 38 nodes
classifier.add(Dense(output_dim = input_nodes, activation='relu'))                          # third layer -> fourth layer 38 nodes
classifier.add(Dense(output_dim = input_nodes, activation='relu'))                          # fourth layer -> output 1 node
classifier.add(Dense(output_dim = output_nodes))                                            # output layer        
sgd = optimizers.SGD(momentum=0.99, lr=0.0001)
classifier.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

history = classifier.fit(X_train,y_train,batch_size=25,epochs=130)
loss, accuracy = classifier.evaluate(X_train,y_train)
print("\nLoss: %.2f, Train accuracy: %.2f%%" % (loss, accuracy*100))

prediction = classifier.predict(X_test)
predictions = [int(x) for x in prediction]
accuracy = np.mean((predictions == y_test-th ) | (predictions == y_test+th) | (predictions == y_test))

print("Test accuracy: %.2f%%" % (accuracy*100))
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
