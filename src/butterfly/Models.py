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
import pyreadr
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import GroupKFold
import butterfly.RF
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from keras import losses
from keras.callbacks import TensorBoard
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
from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects import r, pandas2ri
from sklearn.linear_model import LassoCV
 
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

def NN(albums, DF, feat_n, predictor_index, responses, pixels, 
        features, folds, epochs, optimiser, loss, type_model, 
        type_input, kernel_size, scaler):
    
    groups = DF['patientID']
    
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
            model.add(Dense(features, activation='sigmoid'))
            model.compile(optimizer=optimiser, loss=loss)

            model.fit(X_train, y_train, epochs=epochs, verbose=0)
            
        elif type_model == 'DNN':
            
            model = Sequential()
            model.add(Dense(12, input_dim=X.shape[1], activation='relu'))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            # compile the keras model
            model.compile(loss=loss, optimizer=optimiser)
            # fit the keras model on the dataset
            model.fit(X_train, y_train, epochs=epochs, batch_size=10)

        elif type_model == 'MCNN':
                        
            model = Sequential()
            model.add(Conv2D(filters=64, kernel_size=(kernel_size,kernel_size), activation='relu', input_shape=(X.shape[1], X.shape[2],6)))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Flatten())
            model.add(Dense(50, activation='relu'))
            model.add(Dense(features, activation='sigmoid'))
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

def LRF(DF, responses, predictor_index, feat_n, RF_predictor, folds, ntrees, type_model, type_input, groups_c,scaler):
    
    groups = DF['patientID']
    
    group_cols = groups_c[predictor_index]
    
    #Get your response dataset
    response = responses[predictor_index]
    response_df = DF[response]
    y = response_df.values
    
    yy = y.copy()
    
    if scaler == True:
        y = QuantileTransformer().fit_transform(y)
    
    y = y[:,feat_n]
    
    #Get your predictor dataset
    if type_input == 'TSNE':
        X = np.reshape(RF_predictor[predictor_index], (68,16384))
    elif type_input == 'matrix':
        X = RF_predictor[predictor_index]
                
    #############
    
    results = []
    
    y_prediction_train = []
    y_observed_train = []

    y_prediction_test = []
    y_observed_test = []
    
    group_kfold = GroupKFold(n_splits=folds)
    
    for train_index, test_index in group_kfold.split(X, y, groups):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        yy_train, yy_test = yy[train_index], yy[test_index]
        
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
            
            
        elif type_model == 'Lasso':
        
            model = linear_model.Lasso(alpha=0.1)
            model.fit(X_train, y_train)
            
        elif type_model == 'groupLasso':
        
            model = GLMCV(distr="poisson", 
                          tol=1e-3,
                           group=group_cols, 
                           score_metric="pseudo_R2",
                           alpha=1.0, 
                           learning_rate=3, 
                           max_iter=100, 
                           cv=1, 
                           verbose=True
                         )

            model.fit(X_train, y_train)

        elif type_model == 'sparsegroupLasso':
            
            y_train = y_train.reshape(-1, 1)
        
            model = GroupLasso(
            groups=group_cols,
            group_reg=5,
            l1_reg=0,
            frobenius_lipschitz=True,
            scale_reg="inverse_group_size",
            subsampling_scheme=1,
            supress_warning=True,
            n_iter=1000,
            tol=1e-3,
            )
            model.fit(X_train, y_train)
            
        elif type_model == 'pcLasso':
            
            #nr,nc = X_train.shape
            #X_trainr = ro.r.matrix(X_train, nrow=nr, ncol=nc)
            
            #nr,nc = X_test.shape
            #X_test = ro.r.matrix(X_test, nrow=nr, ncol=nc)
            
            #nr = y_train.shape[0]
            #y_trainr = ro.r.matrix(y_train, nrow=nr, ncol=1)
            
            #robjects.globalenv["X_train"] = X_trainr
            #robjects.globalenv["y_train"] = y_trainr
            
            model = pcLasso.pcLasso(X_train, y_train, ratio=0.8)
            
        elif type_model == 'Dummy':
            
            model = DummyRegressor(strategy="mean")
            model.fit(X_train, y_train)
            
    # demonstrate prediction
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
            
        y_pred_train = pd.DataFrame(y_pred_train)
        y_pred_test = pd.DataFrame(y_pred_test)

        y_train = pd.DataFrame(yy_train)
        y_test  = pd.DataFrame(yy_test)

        y_prediction_train.append(y_pred_train)
        y_observed_train.append(yy_train)
        y_prediction_test.append(y_pred_test)
        y_observed_test.append(yy_test)
    
    return pd.concat(y_prediction_train), pd.concat(y_observed_train), pd.concat(y_prediction_test),pd.concat(y_observed_test)

def RLRF(DF, responses, predictor_index, feat_n, RF_predictor, folds, ntrees, type_model, type_input, groups_c,scaler):
    
    pandas2ri.activate()
    
    groups = DF['patientID']
    
    group_cols = groups_c[predictor_index]
    
    #Get your response dataset
    response = responses[predictor_index]
    response_df = DF[response]
    y = response_df.values
    
    if scaler == True:
        y = QuantileTransformer().fit_transform(y)
    
    y = y[:,feat_n]
    
    #Get your predictor dataset
    if type_input == 'TSNE':
        X = np.reshape(RF_predictor[predictor_index], (68,16384))
    elif type_input == 'matrix':
        X = RF_predictor[predictor_index]
                
    #############
    
    results = []
    
    y_prediction_train = []
    y_observed_train = []

    y_prediction_test = []
    y_observed_test = []
    
    group_kfold = GroupKFold(n_splits=folds)
    
    for train_index, test_index in group_kfold.split(X, y, groups):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        nr, nc = X_train.shape
        X_trainr = ro.r.matrix(X_train, nrow=nr, ncol=nc, byrow=True)

        #X_train_df = pd.DataFrame(X_train)
        #X_trainr = pandas2ri.py2ri(X_train_df)
        
        nr, nc = X_test.shape
        X_testr = ro.r.matrix(X_test, nrow=nr, ncol=nc, byrow=True)

        #X_test_df = pd.DataFrame(X_test)
        #X_testr = pandas2ri.py2ri(X_test_df)

        y_trainr = ro.FloatVector(y_train)

        blocks = np.asarray(groups_c[predictor_index])
    
        blocks_l = np.where(blocks[:-1] != blocks[1:])[0]
        blocks_l = np.append(blocks_l, len(blocks))
        block_list = []
        block = np.array(range(1, blocks_l[0]+1))

        if type_model == 'pcLasso':
                        
            if len(block)>35000:
                block1 = np.array(range(1, 25000))
                block2 = np.array(range(25000, blocks_l[0]+1))
                block_list.append(block1)
                block_list.append(block2)
                
            elif len(block)<35000: 
                block_list.append(block)

            for i in range(len(blocks_l)-1):
                block_list.append(np.array(range(blocks_l[i]+1, blocks_l[i+1]+1)))
                
            blocksr = ro.ListVector([(str(i), x) for i, x in enumerate(block_list)])
            
            ratio = 0.8          
                        
            pcLasso ="""
                function(X_trainr, X_testr, y_trainr, blocksr, ratio){
                    library(pcLasso)         

                    model <- cv.pcLasso(X_trainr, y_trainr, groups = blocksr, ratio = ratio, 
                             verbose = FALSE, nfolds =5, keep = FALSE)

                    predictions <- predict(model, X_testr, s = "lambda.min")
                    predictions
                }"""            

            rfunc=ro.r(pcLasso)
            y_pred_train = rfunc(X_trainr, X_trainr, y_trainr, blocksr, ratio)
            y_pred_test = rfunc(X_trainr, X_testr, y_trainr, blocksr, ratio)
            
            del pcLasso
            
        elif type_model == 'blockForest':
            
            block_list.append(block)

            for i in range(len(blocks_l)-1):
                block_list.append(np.array(range(blocks_l[i]+1, blocks_l[i+1]+1)))
                
            blocksr = ro.ListVector([(str(i), x) for i, x in enumerate(block_list)])
            
            #                colnames(X_trainr) <- paste("X", 1:ncol(X_trainr), sep="")
            
            blockForest ="""
            function(X_trainr, X_testr, y_trainr, blocksr){
                library(blockForest)
                
                colnames(X_trainr) <- paste("X", 1:ncol(X_trainr), sep="")
                colnames(X_testr)  <- paste("X", 1:ncol(X_testr ), sep="")
                                
                forest_obj <- blockfor(X_trainr, y_trainr, num.trees = 100, replace = TRUE, 
                blocks=blocksr, nsets = 10, num.trees.pre = 50, splitrule="extratrees", 
                block.method = "BlockForest")
                                
                prd <- predict(forest_obj$forest, X_testr, block.method = "BlockForest")
                predictions <- predictions(prd)
                
            }"""
            
            rfunc=ro.r(blockForest)
            y_pred_train = rfunc(X_trainr, X_trainr, y_trainr, blocksr)
            y_pred_test = rfunc(X_trainr, X_testr, y_trainr, blocksr)
            
        y_pred_train = np.asarray(y_pred_train)
        y_pred_test = np.asarray(y_pred_test)
                        
        y_pred_train = pd.DataFrame(y_pred_train)
        y_pred_test = pd.DataFrame(y_pred_test)

        y_train = pd.DataFrame(y_train)
        y_test  = pd.DataFrame(y_test)

        y_prediction_train.append(y_pred_train)
        y_observed_train.append(y_train)
        y_prediction_test.append(y_pred_test)
        y_observed_test.append(y_test)
    
    return pd.concat(y_prediction_train), pd.concat(y_observed_train), pd.concat(y_prediction_test),pd.concat(y_observed_test)


def Stacked(DF, omics, responses, predictor_index, feat_n, RF_predictor, folds, ntrees, type_model, metaclassifier, scaler):
        
    groups = DF['patientID']

    #Get your response dataset
    response = responses[predictor_index]
    response_df = DF[response]
    y = response_df.values
    
    if scaler == True:
        y = QuantileTransformer().fit_transform(y)
    
    y = y[:,feat_n]
    
    del omics[predictor_index]
    
    #############
    results = []
    
    y_prediction_train = []
    y_observed_train = []

    y_prediction_test = []
    y_observed_test = []
    
    group_kfold = GroupKFold(n_splits=folds)
    
    X = DF.copy()
                    
    for train_index, test_index in group_kfold.split(X, y, groups):        

        y_pred_train_omic = []
        y_pred_test_omic = []
 
        y_train, y_test = y[train_index], y[test_index]
        groups_train = groups[train_index]
        
        DFB = DF.copy()
        X_train, X_test = DFB.iloc[train_index,:], DFB.iloc[test_index,:]

        for predictor_index in range(len(omics)): 

            #Get your predictor dataset
            wanted = X.columns[X.columns.str.startswith(omics[predictor_index])]

            X_train_o = X_train[wanted].values
            X_test_o = X_test[wanted].values
            
            X_train_o = StandardScaler().fit_transform(X_train_o)                
            X_test_o  = StandardScaler().fit_transform(X_test_o)                
                            
            if type_model == "RF":
                model_o = RandomForestRegressor(n_estimators=100,
                                                min_samples_split = 5,
                                                random_state=0)
                
            elif type_model == 'Lasso':
                model_o = linear_model.Lasso(alpha=0.1)
                
            model_o.fit(X_train_o, y_train)
            
            y_pred_train_omic.append(model_o.predict(X_train_o))
            y_pred_test_omic.append(model_o.predict(X_test_o))
            
        y_pred_test_train_all = []
        y_obsr_test_train = []

        for train_index_train, test_index_train in group_kfold.split(X_train, y_train, groups_train):

            y_pred_test_omic_train = []
            y_obsr_test_omic_train = []

            y_train_train, y_test_train = y_train[train_index_train], y_train[test_index_train]
            X_train_train, X_test_train = X_train.iloc[train_index_train,:], X_train.iloc[test_index_train,:]

            DFB = DF.copy()

            for predictor_index in range(len(omics)): 

                #Get your predictor dataset
                wanted_train = DFB.columns[DFB.columns.str.startswith(omics[predictor_index])]
                
                X_train_o_train = X_train_train[wanted_train].values
                X_test_o_train = X_test_train[wanted_train].values

                X_train_o_train = StandardScaler().fit_transform(X_train_o_train)                
                X_test_o_train  = StandardScaler().fit_transform(X_test_o_train)                

                if type_model == "RF":
                    model_o_train = RandomForestRegressor(n_estimators=100,
                                                          min_samples_split = 5,
                                                          random_state=0)
                    
                elif type_model == 'Lasso':
        
                    model_o_train = linear_model.Lasso(alpha=0.1)

                model_o_train.fit(X_train_o_train, y_train_train)

                y_pred_test_omic_train.append(model_o_train.predict(X_test_o_train))
                y_obsr_test_omic_train.append(y_test_train)
            
            y_pred_test_train = pd.DataFrame(y_pred_test_omic_train).transpose() 
            y_pred_test_train_all.append(y_pred_test_train)
            y_obsr_test_train.append(y_test_train)
            
        y_pred_all_train = pd.concat(y_pred_test_train_all)
        y_obsr_all_train = np.concatenate(y_obsr_test_train, axis=None)
        
        if metaclassifier == "non_negative":

            meta_model = linear_model.Lasso(alpha=0.1, positive = True)

        meta_model.fit(y_pred_all_train, y_obsr_all_train)
            
        y_pred_train = pd.DataFrame(y_pred_train_omic).transpose() 
        X_tr = y_pred_train.values                
            
        y_pred_test = pd.DataFrame(y_pred_test_omic).transpose() 
        X_tst = y_pred_test.values                
                
    # demonstrate prediction
    y_pred_train = meta_model.predict(X_tr)
    y_pred_test = meta_model.predict(X_tst)
            
    y_pred_train = pd.DataFrame(y_pred_train)
    y_pred_test = pd.DataFrame(y_pred_test)

    y_train = pd.DataFrame(y_train)
    y_test  = pd.DataFrame(y_test)

    y_prediction_train.append(y_pred_train)
    y_observed_train.append(y_train)
    y_prediction_test.append(y_pred_test)
    y_observed_test.append(y_test)
    
    return pd.concat(y_prediction_train), pd.concat(y_observed_train), pd.concat(y_prediction_test),pd.concat(y_observed_test)

def Stacked_models(DF, responses, predictor_index, feat_n, RF_predictor, folds, ntrees, type_input, groups_c, metaclassifier, scaler):
    
    groups = DF['patientID']
    
    group_cols = groups_c[predictor_index]
    
    #Get your response dataset
    response = responses[predictor_index]
    response_df = DF[response]
    y = response_df.values
    
    yy = y.copy()
    
    if scaler == True:
        y = QuantileTransformer().fit_transform(y)
    
    y = y[:,feat_n]
    
    #Get your predictor dataset
    if type_input == 'TSNE':
        X = np.reshape(RF_predictor[predictor_index], (68,16384))
    elif type_input == 'matrix':
        X = RF_predictor[predictor_index]
                
    #############
    
    results = []
    
    y_prediction_train = []
    y_observed_train = []

    y_prediction_test = []
    y_observed_test = []
    
    group_kfold = GroupKFold(n_splits=folds)
    
    y_pred_train_models = []
    y_pred_test_models = []
    
    for train_index, test_index in group_kfold.split(X, y, groups):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        yy_train, yy_test = yy[train_index], yy[test_index]
        groups_train = groups[train_index]

        #Model 1
        RF = RandomForestRegressor(n_estimators=ntrees,
                                   min_samples_split = 5,
                                   random_state=0)
        RF.fit(X_train, y_train)
        
        #Model 2
        Lasso = linear_model.Lasso(alpha=0.1)
        Lasso.fit(X_train, y_train)
        
        #Model 3
        y_train_gl = y_train.reshape(-1, 1)
        
        sparsegroupLasso = GroupLasso(
            groups=group_cols,
            group_reg=5,
            l1_reg=0,
            frobenius_lipschitz=True,
            scale_reg="inverse_group_size",
            subsampling_scheme=1,
            supress_warning=True,
            n_iter=1000,
            tol=1e-3,
            )
        sparsegroupLasso.fit(X_train, y_train_gl)
                            
        # demonstrate prediction        
        y_pred_train_RF = pd.DataFrame(RF.predict(X_train))
        y_pred_train_ls = pd.DataFrame(Lasso.predict(X_train))
        y_pred_train_gl = pd.DataFrame(sparsegroupLasso.predict(X_train))
        
        y_pred_train_models.append(y_pred_train_RF)
        y_pred_train_models.append(y_pred_train_ls)
        y_pred_train_models.append(y_pred_train_gl)

        y_pred_test_RF = pd.DataFrame(RF.predict(X_test))
        y_pred_test_ls = pd.DataFrame(Lasso.predict(X_test))
        y_pred_test_gl = pd.DataFrame(sparsegroupLasso.predict(X_test))
        
        y_pred_test_models.append(y_pred_test_RF)
        y_pred_test_models.append(y_pred_test_ls)
        y_pred_test_models.append(y_pred_test_gl)
        
        y_pred_test_train_all = []
        y_obsr_test_train = []
            
        for train_train_index, test_train_index in group_kfold.split(X_train, y_train, groups_train):
        
            X_train_train, X_test_train = X_train[train_train_index], X_train[test_train_index]
            y_train_train, y_test_train = y_train[train_train_index], y_train[test_train_index]
            yy_train_train, yy_test_train = yy_train[train_train_index], yy_train[test_train_index]

            y_pred_test_train = []

            #Model 1
            RF = RandomForestRegressor(n_estimators=ntrees,
                                       min_samples_split = 5,
                                       random_state=0)
            RF.fit(X_train_train, y_train_train)

            #Model 2
            Lasso = linear_model.Lasso(alpha=0.1)
            Lasso.fit(X_train_train, y_train_train)

            #Model 3
            y_train_train_gl = y_train_train.reshape(-1, 1)

            sparsegroupLasso = GroupLasso(
                groups=group_cols,
                group_reg=5,
                l1_reg=0,
                frobenius_lipschitz=True,
                scale_reg="inverse_group_size",
                subsampling_scheme=1,
                supress_warning=True,
                n_iter=1000,
                tol=1e-3,
                )
            sparsegroupLasso.fit(X_train_train, y_train_train_gl)
            
            RF_pred = RF.predict(X_test_train)
            LS_pred = Lasso.predict(X_test_train)
            GL_pred = sparsegroupLasso.predict(X_test_train)
            
            y_pred_test_train.append(RF_pred)        
            y_pred_test_train.append(LS_pred)
            y_pred_test_train.append(GL_pred)
            
            y_pred_test_train_cv = pd.DataFrame(y_pred_test_train).transpose() 
            y_pred_test_train_all.append(y_pred_test_train_cv)
            y_obsr_test_train.append(y_test_train)
        
        y_pred_all_train = pd.concat(y_pred_test_train_all)
        y_obsr_all_train = np.concatenate(y_obsr_test_train, axis=None)
        
        if metaclassifier == "non_negative":

            meta_model = linear_model.Lasso(alpha=0.1, positive = True)

        meta_model.fit(y_pred_all_train, y_obsr_all_train)

        
        y_pred_train = pd.DataFrame(y_pred_train_models).transpose() 
        X_tr = y_pred_train.values                
            
        y_pred_test = pd.DataFrame(y_pred_test_models).transpose() 
        X_tst = y_pred_test.values                
                
    # demonstrate prediction
    y_pred_train = pd.DataFrame(meta_model.predict(X_tr))
    y_pred_test = pd.DataFrame(meta_model.predict(X_tst))

    y_train = pd.DataFrame(y_train)
    y_test  = pd.DataFrame(y_test)

    y_prediction_train.append(y_pred_train)
    y_observed_train.append(y_train)
    y_prediction_test.append(y_pred_test)
    y_observed_test.append(y_test)
    
    return pd.concat(y_prediction_train), pd.concat(y_observed_train), pd.concat(y_prediction_test),pd.concat(y_observed_test)
