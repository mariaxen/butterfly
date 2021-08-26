# %%
# * load data
# * evaluation code (cross validation)
# * predict
#   * BMI
# * setup model
# * experiments
#   * lasso
#   * NN
#   * augmentation: 
#     * NN on replicated data
#     * retrain NN on task (e.g. BMI)

# %%
%load_ext autoreload
%autoreload 2

import pathlib
import pyreadr
import pandas as pd
import butterfly.augmentation.src
from collections import defaultdict
from scipy import stats
import sklearn.metrics
import numpy as np
import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(6)

# %%
#Import your data

project_dir = pathlib.Path("/home/mgbckr/Documents/workspaces/nalab-butterfly")
work_dir = project_dir / "work/augmentation"
data_dir = project_dir / "data/augmentation"

PreE = pyreadr.read_r(str(data_dir / 'PreE_clinical.RData'))

#%%

DF = PreE["all_data"]
DF = DF.dropna()
#DF = DF.drop_duplicates()

groups = DF['Patients.x'].astype(str).str[0:6]

DF = DF[DF.columns.drop(list(DF.filter(regex='ID')))]
DF = DF[DF.columns.drop(list(DF.filter(regex='Patient')))]
DF = DF[DF.columns.drop(list(DF.filter(regex='ga')))]

DF = DF.apply(pd.to_numeric, errors='coerce')

# %%
#Define the predictor datasets

omics = ['rna', 'lipid', 'plasma', 'urine', 'somalog', 'microb']

other = DF.filter(regex='other')
clinical = DF.filter(regex='clinical')
# y = pd.concat([clinical, other], axis=1)
y = other["other_BMI"]

# %%
DF = DF[DF.columns.drop(list(DF.filter(regex='clinical')))]
DF = DF[DF.columns.drop(list(DF.filter(regex='other')))]

# %% example: multi-index
# y.columns = pd.MultiIndex.from_tuples([c.split("_") for c in y.columns])
# y.clinical

# %% LASSO
##########################################################

type_model = 'Lasso'
folds = 10

prediction_train, observed_train, prediction_test, observed_test = \
    butterfly.augmentation.src.LRF(DF, y, folds, type_model, groups)

# %%
spearman_lasso = stats.spearmanr(observed_test, prediction_test)
print(spearman_lasso)

r2_lasso = sklearn.metrics.r2_score(observed_test, prediction_test)
print(r2_lasso)

rmse_lasso = np.sqrt(sklearn.metrics.mean_squared_error(observed_test, prediction_test))
print(rmse_lasso)

mae_lasso = np.sqrt(sklearn.metrics.mean_absolute_error(observed_test, prediction_test))
print(mae_lasso)

# %%

# %% Neural Network
##########################################################

type_model = 'DNN'
folds = 10
epochs = 50 
optimiser = 'adam'
loss = 'mse'

prediction_train, observed_train, prediction_test, observed_test = \
    butterfly.augmentation.src.LRF(DF, y, folds, type_model, groups, 
    epochs, optimiser, loss)

# %%
spearman_DNN = stats.spearmanr(observed_test, prediction_test)
print(spearman_DNN)

r2_DNN = sklearn.metrics.r2_score(observed_test, prediction_test)
print(r2_DNN)

rmse_DNN = np.sqrt(sklearn.metrics.mean_squared_error(observed_test, prediction_test))
print(rmse_DNN)

mae_DNN = np.sqrt(sklearn.metrics.mean_absolute_error(observed_test, prediction_test))
print(mae_DNN)


# %% Augmented Neural Network
##########################################################

type_model = 'DNN'
folds = 10
epochs = 10
optimiser = 'adam'
loss = 'mse'

n_targets = 10
target_omic = "somalog"

# %%
# create augmented dataset
# DF2 = []
# for i in range(100):
#     DF2.append(DF.copy())
# DF2 = pd.concat(DF2)
DF2 = pd.concat([DF.copy() for _ in range(n_targets)])
DF2["target"] = np.repeat(np.arange(77), n_targets)

# %%
# target_omic_data = DF2.filter(regex=target_omic)
# target_omic_indices = np.random.choice(np.arange(target_omic_data.shape[0]), n_targets)

# target_omic_values = target_omic_data.iloc[:, target_indices]
# y = target_omic_values.melt()

# %%
# from random import sample
# responses = np.random.sample([col for col in DF if col.startswith('somalog')], n_targets)
# DF[responses].melt()

# %%
responses = np.random.choice([col for col in DF if col.startswith('somalog')], n_targets, replace=False)
y2 = DF[responses].melt()["value"]
groups2 = np.tile(groups, n_targets)

# %%
# train the augmented neural network

prediction_train, observed_train, prediction_test, observed_test = \
    butterfly.augmentation.src.LRF(DF2, y2, folds, type_model, groups2, 
    epochs, optimiser, loss, work_dir=work_dir)

# %%
# Next steps:
# * save save folds (train_idx and test_idx)
# * save models for each fold
# * retrain for each fold
# * OR put everything into LRF; the MONSTER FUNCTION :P

# %%
#Refit the model
prediction_train, observed_train, prediction_test, observed_test = \
    butterfly.augmentation.src.LRF(DF, y2, folds, type_model, groups2, 
    epochs, optimiser, loss, work_dir=work_dir)

# Evaluation
#########################################################
# %%
spearman = stats.spearmanr(observed_test, prediction_test)
print(spearman)

r2 = sklearn.metrics.r2_score(observed_test, prediction_test)
print(r2)

rmse = np.sqrt(sklearn.metrics.mean_squared_error(observed_test, prediction_test))
print(rmse)

mae = np.sqrt(sklearn.metrics.mean_absolute_error(observed_test, prediction_test))
print(mae)


