# %% enable auto reloading packages
%load_ext autoreload
%autoreload 2

# %% imports
import pathlib
import pickle
import butterfly.picturebook>LRF

# %% settings

# data_path = pathlib.Path("/home/mgbckr/Documents/workspaces/nalab-butterfly")
data_path = pathlib.Path("/home/mgbckr/mnt/nalab/workspaces/nalab-butterfly/data")
out_path = data_path / "_interim/picturebook"

# %% load data / CSV
with open(out_path / "multiomics_training.pkl", "rb") as f:
    data_multiomics_training = pickle.load(f)


# %%

X = data_multiomics_training.drop(columns="meta")
y = data_multiomics_training.meta["gestational_age"]
groups = data_multiomics_training.meta["Gates ID"]

ntrees = 100
type_model = 'Lasso'
scaler = False
longitudinal = True

# %%

butterfly.picturebook.LRF.LRF(X, y, folds, ntrees, type_model, 
groups, scaler,longitudinal)

# %%
