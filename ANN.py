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
from sklearn.cross_decomposition import train_test_split


data = pd.read_csv('data',' ')
# Inputs and otuput of the model
X = data.iloc[:,0:42].values
Y = data.iloc[:,42:45].values


# Data preparation, creating dummy variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X0 = LabelEncoder()
X[:,0] = labelencoder_X0.fit_transform(X[:, 0])

labelencoder_X1 = LabelEncoder()
X[:,1] = labelencoder_X1.fit_transform(X[:, 0])









