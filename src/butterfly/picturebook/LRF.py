import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn import model_selection
from sklearn import linear_model
from sklearn import ensemble
from sklearn import dummy

def LRF(
        X, y, folds, ntrees, type_model, 
        groups, scaler,longitudinal):
            
    #Get your response dataset
    
    if scaler == True:
        y = QuantileTransformer(output_distribution='uniform').fit_transform(y)
                        
    #############
    
    results = []
    
    y_prediction_train = []
    y_observed_train = []

    y_prediction_test = []
    y_observed_test = []
    
    group_kfold = model_selection.GroupKFold(n_splits=folds)
    
    if longitudinal == False:
        
        splitting = model_selection.RepeatedKFold(n_splits=folds, n_repeats=1).split(X)
    
    elif longitudinal == True:
        
        splitting = group_kfold.split(X, y, groups)
    
    for train_index, test_index in splitting:
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        #Train your multioutput RF
            
        if type_model == 'RF':
            model = ensemble.RandomForestRegressor()
            
        elif type_model == 'Lasso':
        
            model = linear_model.Lasso()
            
        elif type_model == 'CV_Lasso':
        
            model = linear_model.LassoCV(
                n_jobs = 1, selection = 'random', 
                n_alphas = 30, cv = 4, max_iter = 100)
            
        elif type_model == 'CV_EN':
        
            model = model = linear_model.ElasticNetCV(
                n_jobs = 1, selection = 'random', 
                n_alphas = 30, cv = 4, max_iter = 100)
            
        elif type_model == 'EN':
        
            model = linear_model.ElasticNet()
                        
        elif type_model == 'gradientBoosting': 
            
            params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
                      'learning_rate': 0.01, 'loss': 'ls'}
            model = ensemble.GradientBoostingRegressor(**params)
            
        elif type_model == 'xgboost': 
            
            model = ensemble.XGBClassifier()
                        
        elif type_model == 'Dummy':
            
            model = dummy.DummyRegressor(strategy="mean")
        
        model.fit(X_train, y_train)
            
    # demonstrate prediction
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
            
        y_pred_train = pd.DataFrame(y_pred_train)
        y_pred_test = pd.DataFrame(y_pred_test)

        y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)

        y_prediction_train.append(y_pred_train)
        y_observed_train.append(y_train)
        y_prediction_test.append(y_pred_test)
        y_observed_test.append(y_test)
    
    return \
        pd.concat(y_prediction_train), \
        pd.concat(y_observed_train), \
        pd.concat(y_prediction_test),\
        pd.concat(y_observed_test)