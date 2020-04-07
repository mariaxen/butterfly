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
import butterfly.RF
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from keras import losses
from keras.callbacks import TensorBoard
from random import sample
 
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

def CNN(albums, DF, feat_n, predictor_index, groups, responses,
        pixels, features, folds, epochs, optimiser, loss, type_model):
    
    groups = DF['patientID']
    
    #Get your response dataset
    response = responses[predictor_index]
    response_df = DF[response]
    y = response_df.values
    y = y[:,feat_n]

    #############
    results = []
    
    y_prediction_train = []
    y_observed_train = []

    y_prediction_test = []
    y_observed_test = []
    
    group_kfold = GroupKFold(n_splits=folds)
    
    if (type_model == "CNN"):
        #CNN
        X = np.asarray(albums[predictor_index])
    
    elif (type_model == "MCNN"):
        #Multi-layered CNN 
        X = [albums[0], albums[1], albums[2], albums[3], albums[4], albums[5], albums[6]]
        del X[predictor_index]
        X = np.array(X, dtype = float)

        X = X.reshape((X.shape[1], pixels, pixels, X.shape[0]))

    for train_index, test_index in group_kfold.split(X, y, groups):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #Train your multioutput RF
    
        if type_model == 'CNN':

            model = Sequential()
            model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(pixels, pixels)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(50, activation='relu'))
            model.add(Dense(features))
            model.compile(optimizer=optimiser, loss=loss)

            model.fit(X_train, y_train, epochs=epochs, verbose=0)
            
        elif type_model == 'MCNN':
                        
            model = Sequential()
            model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu', input_shape=(pixels, pixels,6)))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Flatten())
            model.add(Dense(50, activation='relu'))
            model.add(Dense(features))
            model.compile(optimizer=optimiser, loss=loss)
            
            #        model = Sequential()
            #        model.add(Conv2D(32,(3,3), input_shape=(pixels, pixels,6)))
            #        model.add(Activation('relu'))
            #        model.add(MaxPooling2D(pool_size=(2,2)))
            #        model.add(Flatten())
            #        model.add(Dense(64))
            #        model.add(Dropout(0.2))
            #        model.add(Activation('relu'))
            #        model.add(Dense(features))
            #        model.add(Activation( 'sigmoid'))
            #        model.compile(optimizer='adam', loss='mse')

            model.fit(X_train, y_train, epochs=epochs, verbose=0)

    # demonstrate prediction
        y_pred_train = model.predict(X_train)
        y_pred_train = pd.DataFrame(y_pred_train)

        y_pred_test = model.predict(X_test)
        y_pred_test = pd.DataFrame(y_pred_test)

        y_train = pd.DataFrame(y_train)
        y_test  = pd.DataFrame(y_test)

        y_prediction_train.append(y_pred_train)
        y_observed_train.append(y_train)
        y_prediction_test.append(y_pred_test)
        y_observed_test.append(y_test)
    
    return pd.concat(y_prediction_train), pd.concat(y_observed_train), pd.concat(y_prediction_test),pd.concat(y_observed_test)

def RF(X, y, groups, folds, ntrees, type_model):
    
    results = []
    
    y_prediction_train = []
    y_observed_train = []

    y_prediction_test = []
    y_observed_test = []
    
    group_kfold = GroupKFold(n_splits=folds)
    
    for train_index, test_index in group_kfold.split(X, y, groups):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #Train your multioutput RF
        if type_model == 'daisy_chain':
            model = butterfly.RF.MultiOutputRF(ntrees).fit(X_train, y_train)
        
        elif type_model == 'multi_RF_regressor':
            model = MultiOutputRegressor(RandomForestRegressor(n_estimators=ntrees,
                                                          min_samples_split = 5,
                                                          random_state=0))
            model.fit(X_train, y_train)
        
        elif type_model == 'RF_regressor':
            model = RandomForestRegressor(n_estimators=ntrees,
                                                          min_samples_split = 5,
                                                          random_state=0)
            model.fit(X_train, y_train)
    
    # demonstrate prediction
        y_pred_train = model.predict(X_train)
        y_pred_train = pd.DataFrame(y_pred_train)

        y_pred_test = model.predict(X_test)
        y_pred_test = pd.DataFrame(y_pred_test)

        y_train = pd.DataFrame(y_train)
        y_test  = pd.DataFrame(y_test)

        y_prediction_train.append(y_pred_train)
        y_observed_train.append(y_train)
        y_prediction_test.append(y_pred_test)
        y_observed_test.append(y_test)
    
    return pd.concat(y_prediction_train), pd.concat(y_observed_train), pd.concat(y_prediction_test),pd.concat(y_observed_test)