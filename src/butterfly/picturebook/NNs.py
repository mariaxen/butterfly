from sklearn import preprocessing
from sklearn import model_selection
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from tensorflow.keras import Sequential
import tensorflow.keras.callbacks


def NN(X, y, pixels, folds, epochs, optimiser, loss, type_model, 
        type_input, kernel_size, groups, scaler):
    
    #Get your response dataset
    
    if scaler == True:
        y = preprocessing.QuantileTransformer().fit_transform(y)
    
    #############
    results = []
    
    y_prediction_train = []
    y_observed_train = []

    y_prediction_test = []
    y_observed_test = []
    
    group_kfold = model_selection.GroupKFold(n_splits=folds)

    #Use this for DNN
    if (type_input == 'matrix'):
    #    X = np.reshape(np.asarray(X), (X.shape[0], X.shape[1]))
        X = X

        #Get your predictor dataset
    elif (type_input == "TSNE_S"):
        #For single CNN
        X = np.asarray(X)
        dimensions = 2
    
    elif (type_input == "TSNE_M"):
        #Multi-layered CNN 
        X = np.array(X, dtype = float)
        X = X.reshape((X.shape[1], pixels, pixels, X.shape[0]))
        dimensions = 2
        
    elif (type_input == "TSNE_vgg"):
        #Multi-layered CNN 
        X = [albums, albums2, albums3]
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
            model.add(Dense(1, activation='linear'))
            
            model.compile(optimizer=optimiser, loss=loss)

            model.fit(X_train, y_train, epochs=epochs, verbose=0)
            
        elif type_model == "SimpleCNN":

            callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

            model = Sequential()
            # model.add(
            #     Conv2D(
            #         filters=10,
            #         kernel_size=(kernel_size, kernel_size),
            #         activation='relu', 
            #         input_shape=(X.shape[1], X.shape[2], 7)))
            # model.add(MaxPooling2D())
            # model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(50, activation='relu'))
            model.add(Dense(1, activation='linear'))
            # compile the keras model
            model.compile(loss=loss, optimizer=optimiser)

            # fit the keras model on the dataset
            model.fit(
                X_train, 
                y_train,
                # validation_split=0.2, 
                epochs=epochs, 
                batch_size=10, 
                # callbacks=[callback]
            )

        elif type_model == 'DNN':
            
            model = Sequential()
            model.add(Dense(12, input_dim=X.shape[1], activation='relu'))
            model.add(Dense(8, activation='sigmoid'))
            model.add(Dense(1, activation='linear'))
            # compile the keras model
            model.compile(loss=loss, optimizer=optimiser)
            # fit the keras model on the dataset
            model.fit(X_train, y_train, epochs=epochs, batch_size=10)

        elif type_model == 'SimpleDNN':
            
            model = Sequential()
            model.add(Dense(50, activation='relu'))
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
            model.add(Conv2D(
                filters=64, 
                kernel_size=(kernel_size,kernel_size), 
                activation='relu', 
                input_shape=(X.shape[1], X.shape[2],7)))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Flatten())
            model.add(Dense(50, activation='relu'))
            model.add(Dense(1, activation='linear'))
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
    
    return \
        pd.concat(y_prediction_train), \
        pd.concat(y_observed_train), \
        pd.concat(y_prediction_test),\
        pd.concat(y_observed_test)