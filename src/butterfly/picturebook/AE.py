from numpy import array
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import GroupKFold
import butterfly.RF
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from random import sample
from sklearn import linear_model
from sklearn.dummy import DummyRegressor
import sys, getopt
import re
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import regularizers
from sklearn import ensemble
from sklearn import linear_model
from sklearn.preprocessing import RobustScaler
import kerastuner
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
import pathlib
import sklearn.preprocessing
import kerastuner
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.feature_selection import VarianceThreshold
import tensorflow as tf


def fancy(omics, y, Single_Omics, type_model, folds, groups, scaler, longitudinal, model_kwargs=None):


    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)

    from tensorflow.keras import backend as K

    if model_kwargs is None:
        model_kwargs = dict()

    if scaler == True:
        y = preprocessing.QuantileTransformer().fit_transform(y)
    
    X = Single_Omics
    y = y
    #############    
    y_prediction_train = []
    y_observed_train = []

    y_prediction_test = []
    y_observed_test = []
    
    group_kfold = GroupKFold(n_splits=folds)
            
    if type_model == 'AE':
            
        ncol = []
        input_dim = []
        encoded = []
        encoding_dim = []
        
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.autograph.set_verbosity(0)

        for i in range(len(omics)):
            ncol.append(X[i].shape[1])
            input_dim.append(Input(shape = (ncol[i], ), name = omics[i]))

            encoding_dim.append(50)
            dropout = Dropout(0.2, name = "Dropout" + omics[i])(input_dim[i])

            encoded.append(Dense(encoding_dim[i], activation = 'elu', 
                                 activity_regularizer=regularizers.l1(10e-5), 
                             name = "Encoder"+omics[i])(dropout))
    
        merge = concatenate(encoded)

        bottleneck = Dense(50, kernel_initializer = 'uniform', activation = 'linear', 
                           name = "Bottleneck")(merge)

        merge_inverse = Dense(sum(encoding_dim), 
                              activation = 'elu', name = "Concatenate_Inverse")(bottleneck)

        decoded = []
        for i in range(len(omics)):
                decoded.append(Dense(ncol[i], activation = 'sigmoid', 
                        name = "Decoder" + omics[i])(merge_inverse))

        autoencoder = Model(input_dim, decoded)

        # Compile Autoencoder
        autoencoder.compile(optimizer = 'adam', 
                            loss= {"Decoder" + omics[i]: 'mean_squared_error' for i in range(len(omics))})

        # Autoencoder training
        autoencoder.fit(X, X, epochs = 100, 
                                    batch_size = 16, validation_split = 0.2, 
                                    shuffle = True, verbose = 0)

        # Encoder model
        encoder = Model(input_dim, bottleneck)
        z_merged = pd.DataFrame(encoder.predict(X))

       
    if type_model == 'AE_plaything':
            
        bottleneck_size = model_kwargs.get("bottleneck_size", 10)
        omic_compression_size = model_kwargs.get("omic_compression_size", [30 for _ in range(len(X))])

        ncol = []
        input_dim = []
        encoded = []
        encoding_dim = []
        
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        from tensorflow.keras import backend as K

        for i in range(len(omics)):
            ncol.append(X[i].shape[1])
            input_dim.append(Input(shape = (ncol[i], ), name = omics[i]))

            encoding_dim.append(omic_compression_size[i])
            dropout = Dropout(0.2, name = "Dropout" + omics[i])(input_dim[i])

            encoded.append(Dense(encoding_dim[i], activation = 'elu', 
                                 activity_regularizer=regularizers.l1(10e-5), 
                             name = "Encoder"+omics[i])(dropout))
    
        merge = concatenate(encoded)

        bottleneck = Dense(
            bottleneck_size, 
            kernel_initializer = 'uniform', 
            activation = 'linear', 
            name = "Bottleneck")(merge)

        merge_inverse = Dense(sum(encoding_dim), 
                              activation = 'elu', name = "Concatenate_Inverse")(bottleneck)

        decoded = []
        for i in range(len(omics)):
                decoded.append(Dense(ncol[i], activation = 'sigmoid', 
                        name = "Decoder" + omics[i])(merge_inverse))

        autoencoder = Model(input_dim, decoded)

        # Compile Autoencoder
        autoencoder.compile(optimizer = 'adam', 
                            loss= {"Decoder" + omics[i]: 'mean_squared_error' for i in range(len(omics))})

        # Autoencoder training
        autoencoder.fit(X, X, epochs = 100, 
                                    batch_size = 16, validation_split = 0.2, 
                                    shuffle = True, verbose = 0)

        # Encoder model
        encoder = Model(input_dim, bottleneck)
        z_merged = pd.DataFrame(encoder.predict(X))


    if type_model == 'opt_AE':
            
        ncol = []
        input_dim = []
        encoded = []
        encoding_dim = []
        
        tf.config.threading.set_inter_op_parallelism_threads(1)

        class Butterfly(HyperModel):

            def __init__(self, omic_names, omic_dims):
                self.omic_names = omic_names
                self.omic_dims = omic_dims

            def build(self, hp: kerastuner.HyperParameters):

                input_layers = []

                encoding_layers = []
                encoding_layers_dims = []

                decoding_layers = []
                decoding_layers_names = []

                variables = dict()

                # ENCODING
                for i, (omic_name, omic_dim) in enumerate(zip(self.omic_names, self.omic_dims)):

                    variables[omic_name] = dict()

                    # input
                    input_layer = Input(shape=(omic_dim, ), name=omic_name)
                    input_layers.append(input_layer)

                    # dropout
                    dropout_rate = hp.Float(
                            f"var_dropout_rate_{omic_name}",
                            min_value=0.0,
                            max_value=0.5,
                            default=0.25,
                            step=0.05
                        )

                    dropout_layer = Dropout(
                        rate=dropout_rate, 
                        name=f"dropout_{omic_name}")(input_layer)

                    # number of encoding layers
                    n_layers = hp.Int(f'var_n_layers_{omic_name}', min_value=1, max_value=2)

                    # define encoding layer stack
                    variables[omic_name]["dense_dims"] = []

                    previous_layer = dropout_layer
                    previous_layer_dim = 50

                    for i_layer in range(n_layers):

                        dense_dim = hp.Int(
                            f'var_encode_dense_{omic_name}_{i_layer}', 
                            min_value=5, 
                            max_value=previous_layer_dim)

                        dense = Dense(
                            dense_dim, 
                            activation=hp.Choice('dense_activation',
                            values=['relu', 'tanh', 'sigmoid']),
                            name=f'encode_dense_{omic_name}_{i_layer}')(previous_layer)

                        previous_layer = dense
                        previous_layer_dim = dense_dim
                        variables[omic_name]["dense_dims"].append(dense_dim)

                    encoding_layers.append(previous_layer)
                    encoding_layers_dims.append(previous_layer_dim)

                # Merging Encoder layers from different OMICs
                merge = concatenate(encoding_layers)

                bottleneck_dim = hp.Int(
                    f'var_bottleneck_dim', 
                    min_value=5, 
                    max_value=50)

                # Bottleneck compression
                bottleneck = Dense(
                    bottleneck_dim,
                    kernel_initializer='uniform',
                    activation=hp.Choice('dense_activation',
                    values=['relu', 'tanh', 'sigmoid']),
                    name="Bottleneck")(merge)

                # DECODING LOOP

                # Inverse merging
                merge_inverse = Dense(
                    sum(encoding_layers_dims),
                    activation=hp.Choice('dense_activation',
                    values=['relu', 'tanh', 'sigmoid']), 
                    name="Concatenate_Inverse")(bottleneck)

                for i, (omic_name, omic_dim) in enumerate(zip(self.omic_names, self.omic_dims)):

                    # define encoding layer stack
                    dense_layers_dims = variables[omic_name]["dense_dims"]

                    previous_layer = merge_inverse
                    previous_layer_name = None

                    for i_layer, dens_dim in reversed(list(enumerate(dense_layers_dims))):

                        dens_name = f'decode_dense_{omic_name}_{i_layer}'

                        dens = Dense(
                            dens_dim, 
                            activation=hp.Choice('dense_activation',
                            values=['relu', 'tanh', 'sigmoid']),
                            name=dens_name)(previous_layer)

                        previous_layer_name = dens_name
                        previous_layer = dens

                    last_name = f'last_dense_{omic_name}_{i_layer}'
                    last_layer = Dense(
                            omic_dim, 
                    activation=hp.Choice('dense_activation',
                    values=['relu', 'tanh', 'sigmoid']),
                            name=last_name)(previous_layer)

                    decoding_layers.append(last_layer)
                    decoding_layers_names.append(last_name)

                # Combining Encoder and Decoder into an Autoencoder model
                autoencoder = Model(input_layers, decoding_layers, name="autoencoder")

                # Compile Autoencoder
                autoencoder.compile(
                    optimizer='adam',
                    loss={n : 'mean_squared_error' for n in decoding_layers_names})

                return autoencoder

        # %%
        hypermodel = Butterfly(
            omic_names=omics, 
            omic_dims=[o.shape[1] for o in Single_Omics])
        
        HYPERBAND_MAX_EPOCHS = 20
        MAX_TRIALS = 20
        EXECUTION_PER_TRIAL = 1
        N_EPOCH_SEARCH = 10
        hyperband_iterations = 10

        tuner = Hyperband(
            hypermodel,
            objective='val_loss',
            max_epochs=HYPERBAND_MAX_EPOCHS,
            hyperband_iterations = hyperband_iterations,
            executions_per_trial=EXECUTION_PER_TRIAL,
            directory='hyperband',
            project_name='AE',
            overwrite=True)
        
        tuner.search(Single_Omics,
                    Single_Omics,
                    validation_split=0.15,
                    epochs=N_EPOCH_SEARCH,
                    batch_size = 16, 
                    shuffle = True, 
                    verbose = 0)
        
        autoencoder = tuner.get_best_models(num_models=1)[0]

        # Encoder model
        input_layers = []

        for i in range(len(omics)):

            input_layers.append(autoencoder.get_layer(omics[i]).input)

        bottleneck = autoencoder.get_layer("Bottleneck").output

        encoder = Model(input_layers, bottleneck)

        # Encoder model
        z_merged = pd.DataFrame(encoder.predict(X))

    if longitudinal == False:
        
        splitting = RepeatedKFold(n_splits=folds, n_repeats=1).split(z_merged)
    
    elif longitudinal == True:
        
        splitting = group_kfold.split(z_merged, y, groups)
    
    for train_index, test_index in splitting:

        X_train = z_merged.iloc[train_index,:]
        X_test  = z_merged.iloc[test_index,:]

        y_train, y_test = y[train_index], y[test_index]

        model = RandomForestRegressor()

        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        y_pred_train = pd.DataFrame(y_pred_train)
        y_pred_test = pd.DataFrame(y_pred_test)

        y_prediction_train.append(y_pred_train)
        y_observed_train.append(pd.DataFrame(y_train))
        y_prediction_test.append(y_pred_test)
        y_observed_test.append(pd.DataFrame(y_test))    
        
    return pd.concat(y_prediction_train), pd.concat(y_observed_train), pd.concat(y_prediction_test),pd.concat(y_observed_test)