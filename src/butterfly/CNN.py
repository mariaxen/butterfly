#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:03:50 2020

@author: maria
"""

# univariate cnn example
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import *    
import os 
import pyreadr
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
 
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def model(X, y, groups, pixels, features, folds):
    
    results = []
    y_prediction = []
    y_testing = []
    
    group_kfold = GroupKFold(n_splits=folds)
    
    for train_index, test_index in group_kfold.split(X, y, groups):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #Create your CNN
    #    model = Sequential()
    #    model.add(Conv1D(32,(3), input_shape=(pixels, pixels)))
    #    model.add(Activation('relu'))
    #    model.add(MaxPooling1D(pool_size=(2)))
    #    model.add(Flatten())
    #    model.add(Dense(50))
    #    model.add(Dropout(0.2))
    #    model.add(Activation('relu'))
    #    model.add(Dense(features))
    #    model.add(Activation( 'sigmoid'))
    #    model.compile(optimizer='adam', loss='mse')
    
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(pixels, pixels)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(features))
        model.compile(optimizer='adam', loss='mse')

        model.fit(X_train, y_train, epochs=300, verbose=0)

    # demonstrate prediction
        y_pred = model.predict(X_test, verbose = 0)
        y_pred = pd.DataFrame(y_pred)
        y_test = pd.DataFrame(y_test)

        y_prediction.append(y_pred)
        y_testing.append(y_test)
    
    return pd.concat(y_prediction), pd.concat(y_testing)

def model_multi(response_df, exclude, cv, album, pixels, features):
    
    yy = response_df.drop(['patientID', 'trimester'], axis =1 ).values

    #Create your calibration and validation datasets
    pt1ex = response_df.index[response_df['patientID'] == exclude[cv][0]].tolist()
    pt2ex = response_df.index[response_df['patientID'] == exclude[cv][1]].tolist()
    ptex = pt1ex+pt2ex

    X_c = []
    X_v = []
        
    #Divide in calibration and validation
    for i in range(6):
        X_cN = np.delete(album[i], ptex, 0)
        X_cN = X_cN.reshape((X_cN.shape[0], X_cN.shape[1], pixels))
        X_c.append(X_cN)   

        X_vN = np.asarray([album[i][k]  for k in ptex])
        X_vN = X_vN.reshape((X_vN.shape[0], X_vN.shape[1], pixels))
        X_v.append(X_vN)   
                
    y_c = np.delete(yy, ptex, 0)
    y_c = pd.DataFrame(StandardScaler().fit_transform(y_c))
    
    X_cc = []

    for i in range(60):
        X_cc.append(np.array((X_c[0][i], X_c[1][i], X_c[2][i], X_c[3][i], X_c[4][i], 
                              X_c[5][i]), dtype=float))
    
    X_cc = np.array(X_cc)
    X_cc = X_cc.reshape((X_cc.shape[0], pixels, pixels, X_cc.shape[1]))
    
    y_v = np.asarray([yy[i] for i in ptex])
    y_v = pd.DataFrame(StandardScaler().fit_transform(y_v))
    
    X_vv = []

    for i in range(8):
        X_vv.append(np.array((X_v[0][i], X_v[1][i], X_v[2][i], X_v[3][i], X_v[4][i], 
                             X_v[5][i]), dtype=float))
    
    X_vv = np.array(X_vv)
    X_vv = X_vv.reshape((X_vv.shape[0], pixels, pixels, X_vv.shape[1]))

    #Create your CNN
    model = Sequential()
    model.add(Conv2D(32,(3,3), input_shape=(pixels, pixels,6)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(features))
    model.add(Activation( 'sigmoid'))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_cc, y_c, epochs=300, verbose=0)

    # demonstrate prediction
    y_pred = model.predict(X_vv, verbose = 0)
    y_pred = pd.DataFrame(y_pred)

    return y_pred, y_v