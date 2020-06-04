# %%
print("test")

# %%
# Import Libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from urllib.request import urlretrieve
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import os 
import pickle
from keras.layers import *    
from tensorflow.keras import regularizers
from keras.models import Model
import datetime

# %%
#Define the predictor datasets
omics = ['rna', 'plasma_l', 'serum_l', 'microb', 'immune', 'metabol', 'plasma_s']

os.chdir('/home/mgbckr/Documents/workspaces/nalab-butterfly/data/autoencoder')
with open('Single_Omics_SS.pkl', 'rb') as f:
    X = pickle.load(f)

# %%
ncol = []
input_dim = []
encoded = []
encoding_dim = []

data = X[0]
DATASET_SIZE, input_shape = data.shape
batch_size = 20
max_epochs = 20

# %%
def compute_kernel(x, y, name):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack(
        [x_size, 1, dim], name=name + 's1')), tf.stack([1, y_size, 1], name=name + 's2'))
    tiled_y = tf.tile(tf.reshape(y, tf.stack(
        [1, y_size, dim], name=name + 's3')), tf.stack([x_size, 1, 1], name=name + 's4'))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y),
                                  axis=2) / tf.cast(dim, tf.float32))

def compute_mmd(x, y, name):
    x_kernel = compute_kernel(x, x, name+'x')
    y_kernel = compute_kernel(y, y, name+'y')
    xy_kernel = compute_kernel(x, y, name+'xy')
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - \
        2 * tf.reduce_mean(xy_kernel)

def cust_loss(encoder_input_, encoder_output_, decoder_output_, latent_dim):
    true_samples = tf.random.normal(tf.stack([batch_size, latent_dim]))
    loss_mmd = compute_mmd(true_samples, encoder_output_,'1')
    loss_nll = tf.reduce_mean(tf.square(encoder_input_ - decoder_output_))
    loss = tf.reduce_mean(loss_nll + loss_mmd, name='loss')
    return loss

# %%

class MyHyperModel(HyperModel):

    def build(self, hp):
        LATENT_DIM = hp.Int('latent_dim', min_value=2, max_value=12)
        
        N_LAYERS = hp.Int('n_layers', min_value=2, max_value=5)
        ACTIVATION = hp.Choice('activation',['linear','tanh'])
        hidden_dims = []
        
        for i in range(N_LAYERS):
            hidden_dims.append(hp.Int('units_layer_'+str(i), min_value=10, 
                                                             max_value=128 if i==0 else hidden_dims[-1]))

        encoder_input = tf.keras.Input(shape=(input_shape,), name='enc_input')

        x = tf.keras.layers.Dense(
            hidden_dims[0],
            activation='relu')(encoder_input)

        for d in hidden_dims[1:]:
            x = tf.keras.layers.Dense(d)(x)


            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)

        encoder_output = tf.keras.layers.Dense(LATENT_DIM)(x)

        encoder = tf.keras.Model(encoder_input, encoder_output, name='encoder')

        x = tf.keras.layers.Dense(
            hidden_dims[-1], activation='relu')(encoder_output)

        for d in hidden_dims[-2::-1]:
            x = tf.keras.layers.Dense(d)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            
        decoder_output = tf.keras.layers.Dense(input_shape, activation='relu')(x)

        autoencoder = tf.keras.Model(
            encoder_input, decoder_output, name='autoencoder')

        autoencoder.add_loss(
            cust_loss(encoder_input, encoder_output, decoder_output, LATENT_DIM))

        autoencoder.compile(optimizer='adam', 
        loss )
        return autoencoder


# %%

# Construct the BayesianOptimization tuner using the hypermodel class created
bayesian_tuner = BayesianOptimization(
    hypermodel,
    objective='mean_squared_error',
    max_trials=10,
    seed=10,
    project_name='infoVAE_'+datetime.datetime.now().strftime("%Y%m%d-%H%M"))

# Search for the best parameters of the neural network using the contructed Hypberband tuner
bayesian_tuner.search(data,
            epochs = max_epochs,
            validation_split=0.2,
            batch_size = batch_size)


hypermodel = MyHyperModel()
tuner = Hyperband(
    hypermodel,
    objective='val_loss',
    max_epochs=max_epochs,
    directory='keras-tuner',
    project_name='infoVAE_'+datetime.datetime.now().strftime("%Y%m%d-%H%M"))

# %%
tuner.search(data,
            epochs = max_epochs,
            validation_split=0.2,
            batch_size = batch_size)

# %%
BayesianOptimization(
            hypermodel,
            objective='mse',
            max_trials=10,
            seed=42,
            executions_per_trial=2
        )

# %%
