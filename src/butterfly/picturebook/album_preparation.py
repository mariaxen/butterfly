# %% enable auto reloading packages
%load_ext autoreload
%autoreload 2

# %% imports
import pandas as pd
import pathlib
from butterfly.deepinsight.album.album import AlbumTransformer
import cuml.manifold
import sklearn.manifold
import sklearn.preprocessing
import matplotlib.pyplot as plt
import pickle
import os
import seaborn as sns
import numpy as np

# %% settings

# data_path = pathlib.Path("/home/mgbckr/Documents/workspaces/nalab-butterfly")
data_path = pathlib.Path("/home/mgbckr/mnt/nalab/workspaces/nalab-butterfly/data")
out_path = data_path / "_interim/picturebook"

# make output
os.makedirs(out_path, exist_ok=True)

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
# %%time
omics = [
    'cellfree_rna', 'plasma_luminex', 'serum_luminex', 'microbiome',
    'immune_system', 'metabolomics', 'plasma_somalogic']

# %%
albums = dict()
for o in omics:
    print(o)
    album_transformer = AlbumTransformer(
        64, 
        cuml.manifold.UMAP(),
        layers=[len, np.mean, np.min, np.max],
        store_embeddings=True,
        dimension_scaler=sklearn.preprocessing.QuantileTransformer()
    )

    scaler = sklearn.preprocessing.QuantileTransformer(output_distribution="uniform")
#    minmax = sklearn.preprocessing.MinMaxScaler()

    omics_data = data_multiomics_training[o].values
    omics_data = scaler.fit_transform(omics_data)
    # omics_data = minmax.fit_transform(omics_data)
    # omics_data = omics_data[:, np.isnan(omics_data).sum(axis=0) == 0]

    albums[o] = album_transformer.fit_transform(omics_data)

# %%
with open(out_path / "multiomics_training_albums_individual_omics___algorithm_UMAP___scaling_quantile___dim-scaling_quantile___05.pkl", "wb") as f:
    pickle.dump(albums, f)

# %%
for i in range(10):
    albums = dict()
    for o in omics:
        print(o)
        album_transformer = AlbumTransformer(
            64, 
            cuml.manifold.UMAP(),
            layers=[len, np.mean, np.min, np.max],
            store_embeddings=True,
            dimension_scaler=sklearn.preprocessing.QuantileTransformer()
        )

        scaler = sklearn.preprocessing.QuantileTransformer(output_distribution="uniform")
    #    minmax = sklearn.preprocessing.MinMaxScaler()

        omics_data = data_multiomics_training[o].values
        omics_data = scaler.fit_transform(omics_data)
        # omics_data = minmax.fit_transform(omics_data)
        # omics_data = omics_data[:, np.isnan(omics_data).sum(axis=0) == 0]

        albums[o] = album_transformer.fit_transform(omics_data)

    with open(out_path / f"multiomics_training_albums_individual_omics___algorithm_UMAP___scaling_quantile___dim-scaling_quantile___{i:02d}.pkl", "wb") as f:
        pickle.dump(albums, f)

# %% make albums for each omic
# %%time
omics = [
    'cellfree_rna', 'plasma_luminex', 'serum_luminex', 'microbiome',
    'immune_system', 'metabolomics', 'plasma_somalogic']
albums = dict()
for o in omics:
    print(o)
    album_transformer = AlbumTransformer(64, cuml.manifold.UMAP())
    album_transformer.fit(data_multiomics_training[o].values)
    album = album_transformer.transform(data_multiomics_training[o].values)
    albums[o] = album

# %%
with open(out_path / "multiomics_training_albums_individual_omics___algorithm_UMAP.pkl", "wb") as f:
    pickle.dump(albums, f)

# %% make one album for all omics
album_data = data_multiomics_training.drop(columns="meta").values
album_data = sklearn.preprocessing.QuantileTransformer(output_distribution="uniform").fit_transform(album_data)

# album_transformer = AlbumTransformer(64, cuml.manifold.UMAP(n_neighbors=200, min_dist=0.05))
# album_transformer = AlbumTransformer(64, cuml.manifold.TSNE())  # breaks for this dataset
album_transformer = AlbumTransformer(64, cuml.PCA(n_components=2))
# album_transformer = AlbumTransformer(64, cuml.TruncatedSVD(n_components=2))
album = album_transformer.fit_transform(album_data)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(album[10,1,:,:], ax=axes[0])
sns.heatmap(np.log(album[10,1,:,:]), ax=axes[1])

# %% write album to disk

with open(out_path / "multiomics_training_album___algorithm_PCA.pkl", "wb") as f:
    pickle.dump(album, f)

# %% visualize
# sns.heatmap(album[0,1,::])
