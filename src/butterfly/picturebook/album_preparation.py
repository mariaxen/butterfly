# %% enable auto reloading packages
%load_ext autoreload
%autoreload 2

# %% imports
import pandas as pd
import pathlib
from butterfly.deepinsight.album.album import AlbumTransformer
import cuml.manifold
import sklearn.manifold
import matplotlib.pyplot as plt
import pickle
import os
import seaborn as sns

# %% settings

# data_path = pathlib.Path("/home/mgbckr/Documents/workspaces/nalab-butterfly")
data_path = pathlib.Path("/home/mgbckr/mnt/nalab/workspaces/nalab-butterfly/data")

# %% load data / CSV

data_multiomics_training = pd.read_csv(data_path / "picturebook/multiomics_training.csv.gz", index_col=0, header=[0,1])
data_multiomics_training.columns = pd.MultiIndex.from_tuples([
    ("meta", c[0]) if "Unnamed" in c[1] else c for c in data_multiomics_training.columns])

# %% initialize regular album transformer
# %%time
# album_transformer = AlbumTransformer(64, sklearn.manifold.TSNE())
# album_transformer.fit(data_multiomics_training.immune_system.values)

# %% initialize rapid AI album transformer
%%time
album_transformer = AlbumTransformer(64, cuml.manifold.TSNE())
album_transformer.fit(data_multiomics_training.immune_system.values)

# %% transform data
album = album_transformer.transform(data_multiomics_training.immune_system.values)
print(album.shape)

# %% write album to disk
out_path = data_path / "_interim/picturebook"
os.makedirs(out_path, exist_ok=True)

with open(out_path / "multiomics_training_album.pkl", "wb") as f:
    pickle.dump(album, f)

# %% visualize
sns.heatmap(album[0,1,::])
