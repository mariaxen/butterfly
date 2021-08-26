#%%
import pickle
import pathlib

import sklearn.preprocessing


from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Input, Dense, Dropout

from umap import UMAP
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization
import kerastuner
import datetime



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

mm = sklearn.preprocessing.MinMaxScaler()

Single_Omics = [ mm.fit_transform(o) for o in Single_Omics ]


# %%
# variables:
# * encoding dimensions for each omics
# * dropout rate
# * number of dense layers
# * dimensions of the dense layers
# * bottleneck size



class Butterfly(HyperModel):

    def __init__(self, omic_names, omic_dims):
        self.omic_names = omic_names
        self.omic_dims = omic_dims

    def build(self, hp: kerastuner.HyperParameters):

        input_layers = []
        
        encoding_layers = []
        encoding_layers_dims = []

        decoding_layers = []
        decoding_layers_names = []

        variables = dict()

        # ENCODING
        print("+++++++++++++++++")
        print("ENCODING")
        for i, (omic_name, omic_dim) in enumerate(zip(self.omic_names, self.omic_dims)):
            
            print(omic_name)

            variables[omic_name] = dict()

            # input
            input_layer = Input(shape=(omic_dim, ), name=omic_name)
            input_layers.append(input_layer)

            # dropout
            dropout_rate = hp.Float(
                    f"var_dropout_rate_{omic_name}",
                    min_value=0.0,
                    max_value=0.5,
                    default=0.25,
                    step=0.05
                )

            dropout_layer = Dropout(
                rate=dropout_rate, 
                name=f"dropout_{omic_name}")(input_layer)

            # number of encoding layers
            n_layers = hp.Int(f'var_n_layers_{omic_name}', min_value=1, max_value=2)
            
            # define encoding layer stack
            variables[omic_name]["dense_dims"] = []

            previous_layer = dropout_layer
            previous_layer_dim = 50
            
            for i_layer in range(n_layers):
                
                dense_dim = hp.Int(
                    f'var_encode_dense_{omic_name}_{i_layer}', 
                    min_value=5, 
                    max_value=previous_layer_dim)

                dense = Dense(
                    dense_dim, 
                    activation='elu',
                    name=f'encode_dense_{omic_name}_{i_layer}')(previous_layer)

                print(dense_dim)
                
                previous_layer = dense
                previous_layer_dim = dense_dim
                variables[omic_name]["dense_dims"].append(dense_dim)

            encoding_layers.append(previous_layer)
            encoding_layers_dims.append(previous_layer_dim)

        # Merging Encoder layers from different OMICs
        merge = concatenate(encoding_layers)

        bottleneck_dim = hp.Int(
            f'var_bottleneck_dim', 
            min_value=5, 
            max_value=50)

        # Bottleneck compression
        bottleneck = Dense(
            bottleneck_dim,
            kernel_initializer='uniform',
            activation='linear',
            name="Bottleneck")(merge)

        # DECODING LOOP

        # Inverse merging
        merge_inverse = Dense(
            sum(encoding_layers_dims),
            activation='elu', name="Concatenate_Inverse")(bottleneck)

        print("+++++++++++++++++")
        print("DECODING")
        for i, (omic_name, omic_dim) in enumerate(zip(self.omic_names, self.omic_dims)):
            
            print(omic_name)
            
            # define encoding layer stack
            dense_layers_dims = variables[omic_name]["dense_dims"]
            
            previous_layer = merge_inverse
            previous_layer_name = None

            for i_layer, dens_dim in reversed(list(enumerate(dense_layers_dims))):

                print(dens_dim)

                dens_name = f'decode_dense_{omic_name}_{i_layer}'

                print(dens_name)

                dens = Dense(
                    dens_dim, 
                    activation='elu',
                    name=dens_name)(previous_layer)
                
                previous_layer_name = dens_name
                previous_layer = dens

            last_name = f'last_dense_{omic_name}_{i_layer}'
            last_layer = Dense(
                    omic_dim, 
                    activation='linear',
                    name=last_name)(previous_layer)

            decoding_layers.append(last_layer)
            decoding_layers_names.append(last_name)

        # Combining Encoder and Decoder into an Autoencoder model
        autoencoder = Model(input_layers, decoding_layers, name="autoencoder")

        # Compile Autoencoder
        print(decoding_layers_names)

        autoencoder.compile(
            optimizer='adam',
            loss={n : 'mean_squared_error' for n in decoding_layers_names})

        return autoencoder

# %%
hypermodel = Butterfly(
    omic_names=omics, 
    omic_dims=[o.shape[1] for o in Single_Omics])

tuner = Hyperband(
    hypermodel,
    objective='val_loss',
    max_epochs=5,
    directory='keras-tuner',
    project_name='Butterfly_'+datetime.datetime.now().strftime("%Y%m%d-%H%M"))

tuner.search(Single_Omics,
            Single_Omics,
            epochs = 3,
            validation_split=0.2,
            batch_size = 16)




# %%
hypermodel = Butterfly(
    omic_names=omics, 
    omic_dims=[o.shape[1] for o in Single_Omics])

tuner = Hyperband(
    hypermodel,
    objective='val_loss',
    max_epochs=5,
    directory='keras-tuner',
    project_name='Butterfly_'+datetime.datetime.now().strftime("%Y%m%d-%H%M"))

# %%
tuner = BayesianOptimization(
        hypermodel,
        objective='mse',
        max_trials=10,
        seed=42,
        executions_per_trial=2
    )

tuner.search(Single_Omics, Single_Omics, 
epochs=10, validation_split=0.1, verbose=0)

# %%
tuner.search(Single_Omics,
            epochs = 3,
            validation_split=0.2,
            batch_size = 16)


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
