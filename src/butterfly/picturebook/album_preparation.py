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
out_path = data_path / "_interim/picturebook"

# %% load data / CSV

data_multiomics_training = pd.read_csv(
    data_path / "picturebook/multiomics_training.csv.gz", 
    index_col=0, 
    header=[0,1])
data_multiomics_training.columns = pd.MultiIndex.from_tuples([
    ("meta", c[0]) if "Unnamed" in c[1] else c for c in data_multiomics_training.columns])

with open(out_path / "multiomics_training.pkl", "wb") as f:
    pickle.dump(data_multiomics_training, f)

# %% initialize regular album transformer
# %%time
# album_transformer = AlbumTransformer(64, sklearn.manifold.TSNE())
# album_transformer.fit(data_multiomics_training.immune_system.values)

# %% initialize rapid AI album transformer
%%time
omics = [
    'cellfree_rna', 'plasma_luminex', 'serum_luminex', 'microbiome',
    'immune_system', 'metabolomics', 'plasma_somalogic']
albums = dict()
for o in omics:
    print(o)
    album_transformer = AlbumTransformer(64, cuml.manifold.TSNE())
    album_transformer.fit(data_multiomics_training[o].values)
    album = album_transformer.transform(data_multiomics_training[o].values)
    albums[o] = album

# %%
with open(out_path / "multiomics_training_albums_individual_omics.pkl", "wb") as f:
    pickle.dump(albums, f)

# %%
albums_list_of_arrays = [albums[o][:,1,:,:] for o in omics]

# %% write album to disk
os.makedirs(out_path, exist_ok=True)

with open(out_path / "multiomics_training_album.pkl", "wb") as f:
    pickle.dump(album, f)

# %% visualize
sns.heatmap(album[0,1,::])


# %%
