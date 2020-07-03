# %% enable auto reloading packages
%load_ext autoreload
%autoreload 2

# %% imports
import pathlib
import pickle
import butterfly.picturebook.LRF
import butterfly.picturebook.NNs
import numpy as np
import tensorflow as tf
import seaborn as sns

# %%
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)

# %% settings

# data_path = pathlib.Path("/home/mgbckr/Documents/workspaces/nalab-butterfly")
data_path = pathlib.Path("/home/mgbckr/mnt/nalab/workspaces/nalab-butterfly/data")
out_path = data_path / "_interim/picturebook"

# %% load data / CSV
with open(out_path / "multiomics_training.pkl", "rb") as f:
    data_multiomics_training = pickle.load(f)

# %%

X = data_multiomics_training.drop(columns="meta").values
y = data_multiomics_training.meta["gestational_age"].values
groups = data_multiomics_training.meta["Gates ID"].values

#%%
standard = sklearn.preprocessing.StandardScaler()
minmax = sklearn.preprocessing.MinMaxScaler()

X = standard.fit_transform(X)
#X = minmax.fit_transform(X)

# %%

ntrees = 100
type_model = 'Lasso'
scaler = False
longitudinal = True
folds = 10

_, _, prediction_test, observed_test = \
    butterfly.picturebook.LRF.LRF(
        X, y, folds, ntrees, type_model, 
        groups, scaler,longitudinal)

# %%
import scipy.stats
scipy.stats.spearmanr(prediction_test.values, observed_test.values)

# %%
import matplotlib.pyplot as plt
sns.regplot(observed_test.values, prediction_test.values)
plt.xlim([0, 60])
plt.ylim([0, 60])

# %%
# with open(out_path / "multiomics_training_albums_individual_omics.pkl", "rb") as f:
with open(out_path / "multiomics_training_albums_individual_omics___algorithm_UMAP___scaling_quantile.pkl", "rb") as f:
    albums = pickle.load(f)
albums_list_of_arrays = [albums[o][:,1,:,:] for o in sorted(albums.keys())]

# %%
sns.heatmap(albums["cellfree_rna"][20,1,:,:])

# %%
sns.heatmap(np.log(albums["cellfree_rna"][0,1,:,:]))

# %%
sns.heatmap(albums_list_of_arrays[0][0])

# %%
sns.heatmap(np.log(albums_list_of_arrays[1][0]))

# %%
with open("/home/mxenoc/shared/albums_all_50.pkl", 'rb') as f:
    albums_50 = pickle.load(f)
# %%

sns.heatmap(albums_50[0][0])

# %%

sns.heatmap(np.log(albums_50[0][20]))

# %%

scaler=False
folds = 10

X = albums_list_of_arrays
pixels = 64
epochs = 100
optimiser = 'adam'
loss = 'mse'
type_model = 'MCNN'
type_input = "TSNE_M"
kernel_size = 2

_, _, prediction_test, observed_test = \
    butterfly.picturebook.NNs.NN(X, y, pixels, folds, epochs, optimiser, loss, type_model, 
        type_input, kernel_size, groups, scaler)

# %%

import scipy.stats
scipy.stats.spearmanr(prediction_test.values, observed_test.values)






# %%
# with open(out_path / "multiomics_training_albums_individual_omics.pkl", "rb") as f:
with open(out_path / "multiomics_training_album___algorithm_PCA.pkl", "rb") as f:
    albums = pickle.load(f)
albums_list_of_arrays = albums[:,1,:,:]

# %%
sns.heatmap(albums[10,1,:,:])

# %%

scaler=False
folds = 10

X = albums_list_of_arrays
pixels = 64
epochs = 100
optimiser = 'adam'
loss = 'mse'
type_model = 'CNN'
type_input = "TSNE_S"
kernel_size = 2

_, _, prediction_test, observed_test = \
    butterfly.picturebook.NNs.NN(X, y, pixels, folds, epochs, optimiser, loss, type_model, 
        type_input, kernel_size, groups, scaler)

# %%

import scipy.stats
scipy.stats.spearmanr(prediction_test.values, observed_test.values)

