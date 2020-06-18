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

# %% settings

# data_path = pathlib.Path("/home/mgbckr/Documents/workspaces/nalab-butterfly")
data_path = pathlib.Path("/home/mgbckr/mnt/nalab/workspaces/nalab-butterfly/data")

# %%

data_multiomics_training = pd.read_csv(data_path / "picturebook/multiomics_training.csv.gz", index_col=0, header=[0,1])
data_multiomics_training.columns = pd.MultiIndex.from_tuples([
    ("meta", c[0]) if "Unnamed" in c[1] else c for c in data_multiomics_training.columns])

# %%
# %%time
# album_transformer = AlbumTransformer(64, sklearn.manifold.TSNE())
# album_transformer.fit(data_multiomics_training.immune_system.values)

# %% rapid AI album transformer
%%time
album_transformer = AlbumTransformer(64, cuml.manifold.TSNE())
album_transformer.fit(data_multiomics_training.immune_system.values)

# %%
album = album_transformer.transform(data_multiomics_training.immune_system.values)
print(album.shape)

# %%
plt.imshow(album[0,0,::])

# %%
out_path = data_path / "_interim/picturebook"
os.makedirs(exist_ok=True)

with open(out_path / "multiomics_training_album.pkl", "wb") as f:
    pickle.dump(album, f)

# %%
