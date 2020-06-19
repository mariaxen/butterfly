# %% enable auto reloading packages
%load_ext autoreload
%autoreload 2

# %% imports
import pathlib
import pickle
import butterfly.picturebook.LRF
import butterfly.picturebook.NNs
import numpy as np

# %% settings

# data_path = pathlib.Path("/home/mgbckr/Documents/workspaces/nalab-butterfly")
data_path = pathlib.Path("/home/mgbckr/mnt/nalab/workspaces/nalab-butterfly/data")
out_path = data_path / "_interim/picturebook"

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)

# %% load data / CSV
with open(out_path / "multiomics_training.pkl", "rb") as f:
    data_multiomics_training = pickle.load(f)

# %%

X = data_multiomics_training.drop(columns="meta").values
y = data_multiomics_training.meta["gestational_age"].values
groups = data_multiomics_training.meta["Gates ID"].values

ntrees = 100
type_model = 'RF'
scaler = False
longitudinal = True
folds = 10

# %%

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
with open(out_path / "multiomics_training_albums_individual_omics.pkl", "rb") as f:
    albums = pickle.load(f)
albums_list_of_arrays = [albums[o][:,1,:,:] for o in sorted(albums.keys())]

# %%
with open("/home/mxenoc/shared/albums_all_50.pkl", 'rb') as f:
    albums_50 = pickle.load(f)

X = albums_list_of_arrays
pixels = 64
epochs = 10
optimiser = 'adam'
loss = 'mse'
type_model = 'CNN'
type_input = "TSNE_M"
kernel_size = 2 

_, _, prediction_test, observed_test = \
    butterfly.picturebook.NNs.NN(X, y, pixels, folds, epochs, optimiser, loss, type_model, 
        type_input, kernel_size, groups, scaler)

# %%

import scipy.stats
scipy.stats.spearmanr(prediction_test.values, observed_test.values)

# %%
