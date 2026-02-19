# %%
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import random
from math import sqrt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *


from utils.preprocess import batch_norm, batch_norm_inverse
from utils.visuals import vis_one_var_recon
from utils.metrics import lat_weighted_rmse_one_var

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %% [markdown]
# ### Data

# %%
resolution_folder = '56degree'
resolution = '5.625'  
var_num = '5'

data_train = np.load("/mnt/data/sonia/codicast-data/multivar/concat_1940_2015_" + resolution + "_" + var_num + "var.npy")
data_val = np.load("/mnt/data/sonia/codicast-data/multivar/concat_2016_2016_" + resolution + "_" + var_num + "var.npy")
data_test = np.load("/mnt/data/sonia/codicast-data/multivar/concat_2016_2024_" + resolution + "_" + var_num + "var.npy")

# %%
X_train = data_train.transpose((0, 2, 3, 1))
X_val = data_val.transpose((0, 2, 3, 1))
X_test = data_test.transpose((0, 2, 3, 1))

print(X_train.shape, X_val.shape, X_test.shape)

# %% [markdown]
# ### Data normalization

# %%
X_train_norm = batch_norm(X_train, X_train.shape, batch_size=1460)
X_val_norm = batch_norm(X_val, X_val.shape, batch_size=1460)
X_test_norm = batch_norm(X_test, X_test.shape, batch_size=1460)

print(X_train_norm.shape, X_val_norm.shape, X_test_norm.shape)

# %% [markdown]
# ### Model

# %%
def encoder_net(input_shape):
    encoder_inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (2, 2), activation='relu', padding='same', strides=1)(encoder_inputs)
    x = layers.Conv2D(128, (2, 2), activation='relu', padding='same', strides=1)(x)
    x = layers.Conv2D(256, (2, 2), activation='relu', padding='same', strides=1)(x)

    x = layers.Conv2D(512, (2, 2), activation='relu', padding='same', strides=1, name='bottleneck')(x)

    x = layers.Conv2D(256, (2, 2), activation='relu', padding='same', strides=1)(x)
    x = layers.Conv2D(128, (2, 2), activation='relu', padding='same', strides=1)(x)
    x = layers.Conv2D(5, (2, 2), activation='relu', padding='same', strides=1)(x)
    
    encoder = keras.Model(encoder_inputs, [x], name='encoder_net')
    return encoder

# %%
input_shape = (32, 64, 5)
model = encoder_net(input_shape)
model.summary()

# %% [markdown]
# ### Training

# %%
learning_rate = 1e-4
decay_steps = 10000
decay_rate = 0.95

lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
                                                          decay_steps=decay_steps,
                                                          decay_rate=decay_rate
                                                         )

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss='mse')

# model.compile(optimizer='adam',
#     # optimizer=Adam(learning_rate=lr_schedule),
#              loss='mse', 
#             )

es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=20)
mc = ModelCheckpoint('../saved_models/encoder_cnn_56deg_multivar.h5', 
                     monitor='val_loss', 
                     mode='min',
                     save_best_only=True,
                    )

model.fit(X_train_norm, X_train_norm, 
          validation_data=(X_val_norm, X_val_norm),
          epochs=100, 
          batch_size=128,
          # verbose=2,
          shuffle=True,
          callbacks=[es, mc]
         )

# %% [markdown]
# ### Reconstruction test

# %%
saved_model = load_model('../saved_models/encoder_cnn_56deg_multivar.h5', 
                         # custom_objects={'RandomMaskingLayer': RandomMaskingLayer, 'lat_weighted_loss_mse':lat_weighted_loss_mse}
                        )
X_recon_norm = saved_model.predict(X_test_norm)
# X_recon_norm.shape

# %%
X_recon = batch_norm_inverse(X_test, X_recon_norm, X_recon_norm.shape, 1460)

# %%
# `X_min_per_channel` and `X_max_per_channel` are the min and max values per channel
# X_recon = X_recon_norm * (X_max_per_channel - X_min_per_channel) + X_min_per_channel
X_recon.shape

# %% [markdown]
# ### Postprocessing

# %%
Z500_idx = 0
T850 = 1
T2m_idx = -3
U10_idx = -2
V10_idx = -1

resolution = 2.8125*2

# %%
dict = {"Z500":0, "T850":1, "T2m":-3, "U10":-2, "V10":-1}

for var, var_idx in dict.items():
    print(f'{var} RMSE: {lat_weighted_rmse_one_var(X_recon, X_test, var_idx=var_idx, resolution=resolution):.2f}')

# %%


# %% [markdown]
# #### Z500

# %%
vis_one_var_recon(X_recon, X_test, sample_idx=0, var_idx=0)

# %%
lat_weighted_rmse_one_var(X_recon, X_test, var_idx=0, resolution=2.8125*2)

# %% [markdown]
# #### T850

# %%
vis_one_var_recon(X_recon, X_test, sample_idx=0, var_idx=3)

# %%
lat_weighted_rmse_one_var(X_recon, X_test, var_idx=1, resolution=2.8125*2)

# %% [markdown]
# #### U10

# %%
vis_one_var_recon(X_recon, X_test, sample_idx=0, var_idx=-2)

# %%
lat_weighted_rmse_one_var(X_recon, X_test, var_idx=-2, resolution=2.8125*2)

# %% [markdown]
# #### V10

# %%
vis_one_var_recon(X_recon, X_test, sample_idx=0, var_idx=-1)

# %%
lat_weighted_rmse_one_var(X_recon, X_test, var_idx=-1, resolution=2.8125*2)

# %% [markdown]
# #### T2m

# %%
vis_one_var_recon(X_recon, X_test, sample_idx=0, var_idx=-3)

# %%
lat_weighted_rmse_one_var(X_recon, X_test, var_idx=-3, resolution=2.8125*2)

# %%


# %%


# %%


# %%

latitudes = tf.constant([-88.59375, -85.78125, -82.96875, -80.15625, -77.34375, -74.53125,
                      -71.71875, -68.90625, -66.09375, -63.28125, -60.46875, -57.65625,
                      -54.84375, -52.03125, -49.21875, -46.40625, -43.59375, -40.78125,
                      -37.96875, -35.15625, -32.34375, -29.53125, -26.71875, -23.90625,
                      -21.09375, -18.28125, -15.46875, -12.65625, -9.84375, -7.03125,
                      -4.21875, -1.40625, 1.40625, 4.21875, 7.03125, 9.84375,
                      12.65625, 15.46875, 18.28125, 21.09375, 23.90625, 26.71875,
                      29.53125, 32.34375, 35.15625, 37.96875, 40.78125, 43.59375,
                      46.40625, 49.21875, 52.03125, 54.84375, 57.65625, 60.46875,
                      63.28125, 66.09375, 68.90625, 71.71875, 74.53125, 77.34375,
                      80.15625, 82.96875, 85.78125, 88.59375
                     ], dtype=tf.float32)
# degree --> radians
lat_radians = latitudes * (tf.constant(3.141592653589793, dtype=tf.float32) / 180.0)
print(lat_radians.shape)

cosine_lat = tf.math.cos(lat_radians)
print(cosine_lat.shape)

# Normalize weights
L = cosine_lat / tf.reduce_mean(cosine_lat)
print(L.shape)

# %%
# Expand the dimension for compatibility
L_expanded = tf.expand_dims(tf.expand_dims(tf.expand_dims(L, axis=0), axis=2), axis=3)  # (1, H, 1, 1) for (N, H, W, C) 
print(L_expanded.shape)

# %%
# Calculate squared error and apply latitude weighting factor
squared_error = tf.square(X_recon - X_test)
print(squared_error.shape)

weighted_squared_error = squared_error * L_expanded
print(weighted_squared_error.shape)

# %%
# Calculate mean over all dimensions except the channel dimension
weighted_error = tf.reduce_mean(weighted_squared_error)
weighted_error.numpy()


