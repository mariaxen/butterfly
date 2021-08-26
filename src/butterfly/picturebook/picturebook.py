
#%% Preamble

%load_ext autoreload
%autoreload 2

# %%
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import numpy as np
import pandas as pd
import scipy.stats
import pickle

from sklearn.preprocessing import QuantileTransformer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, Flatten

# %%
with open("/home/mxenoc/shared/albums_all_50.pkl", 'rb') as f:
    albums = pickle.load(f)

with open("/home/mxenoc/shared/y_omics.pkl", 'rb') as f:
    y_omics = pickle.load(f)

# %%
with open("/home/mxenoc/shared/prostate.pkl", 'rb') as f:
    prostate = pickle.load(f)

with open("/home/mxenoc/shared/y_prostate.pkl", 'rb') as f:
    y_prostate = pickle.load(f)


# %%
## find non-numeric values 
# stomach_values = prostate.values
# for i in range(stomach_values.shape[1]):
#     for j in range(stomach_values.shape[0]):
#         v = stomach_values[j,i]
#         if isinstance(v, str) and "Annex" in v:
#             print(i, j, v)

# %%
# drop first row because it's weird (contains strings which look like column names?)
prostate = prostate.iloc[1:,:].apply(pd.to_numeric, errors='coerce')
y_prostate = y_prostate.iloc[1:,:]

d_transform = prostate.values[:,-10000:]

# %%
%%time
from butterfly.deepinsight.album2 import AlbumTransformer
album_transfomer = AlbumTransformer(size=64)
album_transfomer.fit(d_transform)

# %%
%time
album = album_transfomer.transform(d_transform)

# %%
import matplotlib.pyplot as plt
plt.imshow(album[0][0])

# %% parallel album
%%time
import sklearn.base
class EmbeddingTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit_transform(self, X):
        import openTSNE
        return openTSNE.TSNE(n_jobs=50).fit(X)

from butterfly.deepinsight.album2 import AlbumTransformer
album_transfomer = AlbumTransformer(
    size=64, embedding_algorithm=EmbeddingTransformer())
album_transfomer.fit(d_transform)

# %%
%time
album = album_transfomer.transform(d_transform)


# %% CNN
# Scale your output 

model.fit(X_train, y_train, epochs=epochs, verbose=0)
y = y_omics['clinical_GA']
y = QuantileTransformer().fit_transform(y.values.reshape(-1,1))

X = np.asarray(albums[0])
dimensions = 2
kernel_size = 2
number_features = y.shape[1]
optimiser = 'adam'
loss = 'mse'
epochs = 25

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, 
#test_size=0.10, random_state=42)

splitting = group_kfold.split(X, y, groups)
    
for train_index, test_index in splitting:
        
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=kernel_size, activation='relu', 
    input_shape=(X.shape[1], X.shape[2])))
    model.add(MaxPooling1D(pool_size=dimensions))

    model.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu'))
    #, input_shape=(X.shape[1], X.shape[2])))
    model.add(MaxPooling1D(pool_size=dimensions))

    model.add(Flatten())

    model.add(Dense(50, activation='relu'))
    model.add(Dense(25, activation='sigmoid'))            
    model.add(Dense(number_features, activation='linear'))

    model.compile(optimizer=optimiser, loss=loss)

    model.fit(X_train, y_train, epochs=epochs, verbose=0)

    CNN_predict = model.predict(X_test)
    y_prediction_test = pd.DataFrame(CNN_predict)
    y_observed_test  = pd.DataFrame(y_test)

    scipy.stats.spearmanr(y_prediction_test, y_observed_test)


# %% Lasso model 
from sklearn import linear_model

X = np.reshape(albums[0], (68,16384))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size=0.10, random_state=42)

lasso_model = linear_model.Lasso()
lasso_model.fit(X_train, y_train)
lasso_predict = lasso_model.predict(X_test)
lasso_predict = pd.DataFrame(lasso_predict)

scipy.stats.spearmanr(lasso_predict, y_observed_test)
