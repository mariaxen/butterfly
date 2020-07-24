

#%%
import sklearn.linear_model
import numpy as np
import scipy.stats
import sklearn.datasets

X, y = sklearn.datasets.make_regression()


#%%
model = sklearn.linear_model.Lasso()
model.fit(X,y)
scipy.stats.spearmanr(model.predict(X), y)

# %%
