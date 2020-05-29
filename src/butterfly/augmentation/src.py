from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LassoCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import pathlib


def LRF(
        DF, 
        y, 
        folds, 
        type_model, 
        groups, 
        epochs=None, 
        optimiser=None, 
        loss=None,
        work_dir="."):
        
    DFB = DF.copy()

    DFB = StandardScaler().fit_transform(DFB.values)

    X = DFB
                    
    #############
    
    results = []
    
    y_prediction_train = []
    y_observed_train = []

    y_prediction_test = []
    y_observed_test = []
    
    group_kfold = GroupKFold(n_splits=folds)
    
    splitting = group_kfold.split(X, y, groups)
    
    for train_index, test_index in splitting:
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if type_model == 'Lasso':
        
            model = LassoCV(n_jobs = 1, selection = 'random', 
                                n_alphas = 10, cv = 4)
                    
            model.fit(X_train, y_train)

        elif type_model == 'DNN':
            
            model = Sequential()
            model.add(Dense(12, input_dim=X.shape[1], activation='relu'))
            model.add(Dense(8, activation='sigmoid'))
            model.add(Dense(1, activation='linear'))
            # compile the keras model
            model.compile(loss=loss, optimizer=optimiser)
            # fit the keras model on the dataset
            model.fit(X_train, y_train, epochs=epochs, batch_size=10)

            model.save(
                str(pathlib.Path(work_dir) / 'MyModel_h5.h5'), 
                save_format='h5')

    # demonstrate prediction
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
            
        y_pred_train = pd.DataFrame(y_pred_train)
        y_pred_test = pd.DataFrame(y_pred_test)

        y_train = pd.DataFrame(y_train)
        y_test  = pd.DataFrame(y_test)

        y_prediction_train.append(y_pred_train)
        y_observed_train.append(y_train)
        y_prediction_test.append(y_pred_test)
        y_observed_test.append(y_test)
    
    return pd.concat(y_prediction_train), pd.concat(y_observed_train), pd.concat(y_prediction_test),pd.concat(y_observed_test)