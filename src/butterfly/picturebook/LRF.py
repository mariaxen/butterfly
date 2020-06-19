import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import GroupKFold


def LRF(DF, responses, predictor_index, feat_n, RF_predictor, folds, ntrees, type_model, type_input, groups, groups_c,scaler,longitudinal):
        
    group_cols = groups_c[predictor_index]
    
    #Get your response dataset
    
    if scaler == True:
        y = QuantileTransformer(output_distribution='uniform').fit_transform(y)
        
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
    
    if longitudinal == False:
        
        splitting = RepeatedKFold(n_splits=folds, n_repeats=1).split(X)
    
    elif longitudinal == True:
        
        splitting = group_kfold.split(X, y, groups)
    
    for train_index, test_index in splitting:
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        yy_train, yy_test = yy[train_index], yy[test_index]
        
        #Train your multioutput RF
            
        if type_model == 'RF':
            model = RandomForestRegressor()
            
        elif type_model == 'Lasso':
        
            import cuml.linear_model
            model = cuml.linear_model.Lasso()
            
        elif type_model == 'CV_Lasso':
        
            model = LassoCV(n_jobs = 1, selection = 'random', 
                            n_alphas = 30, cv = 4, max_iter = 100)
            
        elif type_model == 'CV_EN':
        
            model = model = ElasticNetCV(n_jobs = 1, selection = 'random', 
                            n_alphas = 30, cv = 4, max_iter = 100)
            
        elif type_model == 'EN':
        
            model = ElasticNet()
                        
        elif type_model == 'gradientBoosting': 
            
            params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
                      'learning_rate': 0.01, 'loss': 'ls'}
            model = ensemble.GradientBoostingRegressor(**params)
            
        elif type_model == 'xgboost': 
            
            model = XGBClassifier()

        elif type_model == 'sparsegroupLasso':
            
            y_train = y_train.reshape(-1, 1)
        
            model = GroupLasso(
                groups=group_cols,
                group_reg=5,
                l1_reg=0.0,
                frobenius_lipschitz=True,
                scale_reg="inverse_group_size",
                subsampling_scheme=1,
                supress_warning=True,
                n_iter=100,
                tol=1e-3,
                warm_start = True
            )
            
        elif type_model == 'SparsegroupLasso':
            model = GroupLasso()
                        
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
        y_observed_train.append(y_train)
        y_prediction_test.append(y_pred_test)
        y_observed_test.append(y_test)
    
    return pd.concat(y_prediction_train), pd.concat(y_observed_train), pd.concat(y_prediction_test),pd.concat(y_observed_test)