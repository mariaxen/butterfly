#%%

###
# common imports
###

# general
import re
import collections
import pickle
import pathlib
import warnings

# data
import numpy as np
import pandas as pd

# ml / stats
import sklearn
import statsmodels.stats.multitest

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

# init matplotlib defaults (for Nima)
import matplotlib
matplotlib.rcParams['figure.facecolor'] = 'white'

#%%

from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.preprocessing import StandardScaler
from keras.layers import *    
from keras import Sequential   
import os 
import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
import butterfly.album
from itertools import combinations 
from joblib import parallel_backend
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import r2_score
import pickle
from sklearn.model_selection import GroupKFold
from random import sample
from scipy import stats
from sklearn.metrics import mean_absolute_error
from collections import defaultdict
import time
from keras.applications.resnet50 import ResNet50
from sklearn.dummy import DummyRegressor
import random
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

#%% load data
import pickle
DF = pickle.load(open("../../../data/transfer/DF.pkl", "rb"))
albums = pickle.load(open("../../../data/transfer/albums_all.pkl", "rb"))
responses = pickle.load(open("../../../data/transfer/responses.pkl", "rb"))

#%% prepare data

groups = DF['patientID']

#Get your response dataset
predictor_index = 1
feat_n = 0
folds = 10 #number of folds
features = 1 #number of features to predict
epochs = 200 #number of epochs
optimiser = 'adam' #model optimiser
loss = 'mse' #model loss
ntrees = 100
kernel_size = 2
dimensions = 2
    
#Get your X
X = [albums[predictor_index], albums[predictor_index], albums[predictor_index]]
X = np.array(X, dtype = float)
X = X.reshape((X.shape[1], 128, 128, X.shape[0]))

#Get your y
response = responses[predictor_index]
response_df = DF[response]
y = response_df.values
y = y[:,feat_n]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=groups)

#%% simple CNN (3d)

model = Sequential()
model.add(Conv2D(
    filters=64, 
    kernel_size=(kernel_size,kernel_size), 
    activation='relu', 
    input_shape=(X.shape[1], X.shape[2],3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(features, activation='linear'))
model.compile(optimizer=optimiser, loss=loss)


#%%
model.fit(X_train, y_train, epochs=epochs, verbose=0)

#%%
y_pred_test = model.predict(X_test)

#%%
y_pred_test

#%%
y_test

# %%
import scipy
scipy.stats.spearmanr(y_pred_test, y_test)

#%%
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred_test)


#%%

# Source: https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a
# See: `tf_15.py` and `tf_20.py`

from keras.applications import vgg16
from keras.models import Model
import keras

# parameters

input_shape=(128, 128, 3)

# load and adjust pre-trained model

vgg = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

output = vgg.layers[-1].output
output = keras.layers.Flatten()(output)
vgg_model = Model(vgg.input, output)

vgg_model.trainable = False
for layer in vgg_model.layers:
    layer.trainable = False

# stitch models together

model = Sequential()
model.add(vgg_model)

# original
# model.add(Dense(512, activation='relu', input_dim=input_shape))
# model.add(Dropout(0.3))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1, activation='linear'))

# maria
model.add(Dense(50, activation='relu'))
model.add(Dense(features, activation='linear'))

# compile
model.compile(loss=loss, optimizer=optimiser)

# inspect model
import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])  

#%%
model.fit(X_train, y_train, epochs=epochs, verbose=0)

#%%
y_pred_test = model.predict(X_test)


#%%
y_pred_test

#%%
y_test

# %%
import scipy
scipy.stats.spearmanr(y_pred_test, y_test)

#%%
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred_test)
