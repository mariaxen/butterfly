#%%
import pickle
import pathlib

import sklearn.preprocessing

from keras.models import Model
import matplotlib.pyplot as plt
from keras.layers.merge import concatenate
from keras.layers import Input, Dense, Dropout

from umap import UMAP


# %%
data_dir = pathlib.Path("/home/mgbckr/Documents/workspaces/nalab-butterfly/data/autoencoder")

# %%
# load data

with open(data_dir / 'Single_Omics_SS.pkl', 'rb') as f:
    Single_Omics = pickle.load(f)

omics = [
    'rna', 'plasma_l', 'serum_l', 'microb', 
    'immune', 'metabol', 'plasma_s']

# %%
# scaling

Single_Omics_original = Single_Omics
robust = sklearn.preprocessing.RobustScaler()
mm = sklearn.preprocessing.MinMaxScaler()
Single_Omics = [ mm.fit_transform(robust.fit_transform(o)) for o in Single_Omics ]


# %%

encoding_dim = [30 for _ in range(len(omics))]

n_col = []

input_dims = []
dropout_layers = []
encoding_layers = []
decoding_layers = []

for i in range(len(omics)):
    
    n_col.append(Single_Omics[i].shape[1])

    input_dim = Input(shape=(n_col[i], ), name=omics[i])
    input_dims.append(input_dim)

    dropout_layer = Dropout(0.2, name=f"Dropout_{omics[i]}")(input_dim)
    dropout_layers.append(dropout_layer)

    encoded = Dense(
        encoding_dim[i], 
        activation='elu',
        name=f"Encoder_{omics[i]}")(dropout_layer)
    encoding_layers.append(encoded)

# Merging Encoder layers from different OMICs
merge = concatenate(encoding_layers)

# Bottleneck compression
bottleneck = Dense(
    50, 
    kernel_initializer='uniform',
    activation='linear',
    name="Bottleneck")(merge)

# Inverse merging
merge_inverse = Dense(
    sum(encoding_dim),
    activation='elu', name="Concatenate_Inverse")(bottleneck)

for i in range(len(omics)):
    decoding_layer = Dense(
        n_col[i],
        activation='sigmoid',
        name=f"Decoder_{omics[i]}")(merge_inverse)
    decoding_layers.append(decoding_layer)

# Combining Encoder and Decoder into an Autoencoder model
autoencoder = Model(input=input_dims, output=decoding_layers)

# Compile Autoencoder
autoencoder.compile(
    optimizer='adam',
    loss={f"Decoder_{omics[i]}" : 'mean_squared_error' for i in range(len(omics))})
autoencoder.summary()

# %%

# Autoencoder training
estimator = autoencoder.fit(
    Single_Omics,
    Single_Omics,
    epochs=130,
    batch_size=16,
    validation_split=0.2,
    shuffle=True,
    verbose=1)

# %%

print("Training Loss: ",estimator.history['loss'][-1])
print("Validation Loss: ",estimator.history['val_loss'][-1])
plt.plot(estimator.history['loss']); plt.plot(estimator.history['val_loss'])
plt.title('Model Loss'); plt.ylabel('Loss'); plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc = 'upper right')

# %%

# Encoder model
encoder = Model(input=input_dims, output=bottleneck)
bottleneck_representation = encoder.predict(Single_Omics)

# %%

# ############### UNIFORM MANIFOLD APPROXIMATION AND PROJECTION (UMAP) ###############
model_umap = UMAP(n_neighbors=11, min_dist=0.1, n_components=2)
umap = model_umap.fit_transform(bottleneck_representation)

# %%

plt.scatter(umap[:, 0], umap[:, 1], cmap='tab10', s=10)
plt.title('UMAP on Autoencoder: Data Integration, scNMTseq')
plt.xlabel("UMAP1"); 
plt.ylabel("UMAP2")
