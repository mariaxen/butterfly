# univariate cnn example
from numpy import array
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Flatten
#from keras.layers.convolutional import Conv1D
#from keras.layers.convolutional import Conv2D
#from keras.layers.convolutional import MaxPooling1D
#from keras.layers import *    
import pyreadr
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import GroupKFold
import butterfly.RF
import butterfly.stratified_Kfold
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
#from keras import losses
#from keras.callbacks import TensorBoard
from random import sample
from sklearn import linear_model
from sklearn.dummy import DummyRegressor
from pyglmnet import GLMCV
from group_lasso import GroupLasso
import rpy2.robjects as ro
import sys, getopt
import re
import tzlocal
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
#from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects import r, pandas2ri
from sklearn.linear_model import LassoCV
#from keras.applications import vgg16
#from keras.models import Model
import keras
#from keras import optimizers
from sklearn import ensemble
from xgboost import XGBClassifier
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from sklearn.model_selection import RandomizedSearchCV
import tensorflow as tf
from keras import backend as K
import maui
import maui.utils
import os
import urllib.request
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import maui
import maui.utils
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import regularizers
from sklearn import ensemble
from sklearn import linear_model
from skopt import BayesSearchCV
from sklearn.preprocessing import RobustScaler
from skopt.space import Real, Categorical, Integer
#from keras.layers import *    
#from keras.models import Model
import kerastuner
import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
import pickle
import pathlib
import sklearn.preprocessing
import kerastuner
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression

def NN(albums, albums2, albums3, DF, feat_n, predictor_index, responses, pixels, 
        features, folds, epochs, optimiser, loss, type_model, 
        type_input, kernel_size, groups, scaler):
    
    #Get your response dataset
    response = responses[predictor_index]
    response_df = DF[response]
    y = response_df.values
    
    if scaler == True:
        y = QuantileTransformer().fit_transform(y)
    
    y = y[:,feat_n]

    #############
    results = []
    
    y_prediction_train = []
    y_observed_train = []

    y_prediction_test = []
    y_observed_test = []
    
    group_kfold = GroupKFold(n_splits=folds)
    
        #Get your predictor dataset
        #Use this for CNN
    if type_input == 'matrix_3':
        X = np.reshape(np.asarray(albums[predictor_index]), (albums[predictor_index].shape[0],1,albums[predictor_index].shape[1]))
        dimensions = 1
        
        #Use this for DNN
    elif type_input == 'matrix_2':
        X = np.reshape(np.asarray(albums[predictor_index]),
                        (albums[predictor_index].shape[0], albums[predictor_index].shape[1]))

    elif (type_input == "TSNE_S"):
        #CNN
        X = np.asarray(albums[predictor_index])
        dimensions = 2
    
    elif (type_input == "TSNE_M"):
        #Multi-layered CNN 
        X = [albums[0], albums[1], albums[2], albums[3], albums[4], albums[5], albums[6]]
        del X[predictor_index]
        X = np.array(X, dtype = float)

        X = X.reshape((X.shape[1], pixels, pixels, X.shape[0]))
        dimensions = 2
        
    elif (type_input == "TSNE_vgg"):
        #Multi-layered CNN 
        X = [albums[predictor_index], albums2[predictor_index], albums3[predictor_index]]
        X = np.array(X, dtype = float)
        
        X = X.reshape((X.shape[1], pixels, pixels, X.shape[0]))
        dimensions = 2        

    for train_index, test_index in group_kfold.split(X, y, groups):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #Train your multioutput RF
    
        if type_model == 'CNN':

            model = Sequential()
            model.add(Conv1D(filters=16, kernel_size=kernel_size, activation='relu', input_shape=(X.shape[1], X.shape[2])))
            model.add(MaxPooling1D(pool_size=dimensions))
            
            model.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu'))#, input_shape=(X.shape[1], X.shape[2])))
            model.add(MaxPooling1D(pool_size=dimensions))
            
            model.add(Flatten())
            
            model.add(Dense(50, activation='relu'))
            model.add(Dense(25, activation='sigmoid'))            
            model.add(Dense(features, activation='linear'))
            
            model.compile(optimizer=optimiser, loss=loss)

            model.fit(X_train, y_train, epochs=epochs, verbose=0)
            
        elif type_model == 'DNN':
            
            model = Sequential()
            model.add(Dense(12, input_dim=X.shape[1], activation='relu'))
            model.add(Dense(8, activation='sigmoid'))
            model.add(Dense(1, activation='linear'))
            # compile the keras model
            model.compile(loss=loss, optimizer=optimiser)
            # fit the keras model on the dataset
            model.fit(X_train, y_train, epochs=epochs, batch_size=10)
            
        elif type_model == 'vgg':
            
            input_shape=(pixels, pixels, 3)

            vgg = vgg16.VGG16(include_top=False, weights='imagenet', 
                                                 input_shape=input_shape)

            output = vgg.layers[-1].output
            output = keras.layers.Flatten()(output)
            vgg_model = Model(vgg.input, output)

            vgg_model.trainable = False
            for layer in vgg_model.layers:
                layer.trainable = False

            pd.set_option('max_colwidth', -1)
            layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
            pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])    

            # tf20
            model = Sequential()            
            model.add(vgg_model)
            
            #model.add(Dense(512, activation='relu', input_dim=input_shape))
            #model.add(Dropout(0.3))
            #model.add(Dense(512, activation='relu'))
            #model.add(Dropout(0.3))
            #model.add(Dense(1, activation='linear'))

            #model.add(Dense(50, activation='relu'))
            model.add(Dense(features, activation='linear'))

            model.compile(loss=loss, optimizer=optimiser)
            
            #model.fit(X_train, y_train, epochs=epochs, verbose=0)
            
        elif type_model == 'vgg_small':
            
            input_shape=(pixels, pixels, 3)

            vgg = vgg16.VGG16(include_top=False, weights='imagenet', 
                                                 input_shape=input_shape)

            output = vgg.layers[-1].output
            output = keras.layers.Flatten()(output)
            vgg_model = Model(vgg.input, output)

            vgg_model.trainable = False
            for layer in vgg_model.layers:
                layer.trainable = False

            pd.set_option('max_colwidth', -1)
            layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
            pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])    

            # tf20
            model = Sequential()            
            model.add(vgg_model)            
            model.add(Dense(50, activation='relu'))
            model.add(Dense(features, activation='linear'))

            model.compile(loss=loss, optimizer=optimiser)
            
            model.fit(X_train, y_train, epochs=epochs, verbose=0)
            
        elif type_model == 'vgg_only':
            
            input_shape=(pixels, pixels, 3)

            vgg = vgg16.VGG16(include_top=False, weights='imagenet', 
                                                 input_shape=input_shape)

            output = vgg.layers[-1].output
            output = keras.layers.Flatten()(output)
            vgg_model = Model(vgg.input, output)

            vgg_model.trainable = False
            for layer in vgg_model.layers:
                layer.trainable = False

            pd.set_option('max_colwidth', -1)
            layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
            pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])    

            # tf20
            model = Sequential()            
            model.add(vgg_model)
            model.add(Dense(features, activation='linear'))

            model.compile(loss=loss, optimizer=optimiser)
            
            model.fit(X_train, y_train, epochs=epochs, verbose=0)

        elif type_model == 'MCNN':
                        
            model = Sequential()
            model.add(Conv2D(filters=64, kernel_size=(kernel_size,kernel_size), activation='relu', input_shape=(X.shape[1], X.shape[2],6)))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Flatten())
            model.add(Dense(50, activation='relu'))
            model.add(Dense(features, activation='linear'))
            model.compile(optimizer=optimiser, loss=loss)
            
            #model = Sequential()
            #model.add(Conv2D(32,(3,3), input_shape=(X.shape[1], X.shape[2],6)))
            #model.add(Activation('relu'))
            #model.add(MaxPooling2D(pool_size=(dimensions,dimensions)))
            #model.add(Flatten())
            #model.add(Dense(64))
            #model.add(Dropout(0.2))
            #model.add(Activation('relu'))
            #model.add(Dense(features))
            #model.add(Activation( 'sigmoid'))
            #model.compile(optimizer=optimiser, loss=loss)

            model.fit(X_train, y_train, epochs=epochs, verbose=0)
            
#        elif type_model == 'ResNet':
            
#            model = ResNet50(weights='imagenet')

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

