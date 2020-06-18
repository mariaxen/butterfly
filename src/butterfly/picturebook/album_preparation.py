# %%
import pandas as pd
import pathlib
from butterfly.deepinsight.album.album import AlbumTransformer

# %%

# data_path = pathlib.Path("/home/mgbckr/Documents/workspaces/nalab-butterfly")
data_path = pathlib.Path("/home/mgbckr/mnt/nalab/workspaces/nalab-butterfly/data")

# %%

data_multiomics_training = pd.read_csv(data_path / "picturebook/multiomics_training.csv.gz", index_col=0, header=[0,1])
data_multiomics_training.columns = pd.MultiIndex.from_tuples([
    ("meta", c[0]) if "Unnamed" in c[1] else c for c in data_multiomics_training.columns])


# %%
import cuml.manifold

# %%
album_transformer = AlbumTransformer(64, cuml.manifold.TSNE())

# %%
