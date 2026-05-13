# %%
import os
import sys

# 1. Hardcode your Conda prefix (since we know where it lives on fred-desktop-2)
conda_prefix = '/hdd2/sonia/miniconda3/envs/codicast'

# 2. Force the system to look here for the NVIDIA libraries
os.environ['LD_LIBRARY_PATH'] = (
    f"{conda_prefix}/lib/python3.10/site-packages/nvidia/cudnn/lib:"
    f"{conda_prefix}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:"
    f"{conda_prefix}/lib/python3.10/site-packages/nvidia/cublas/lib:"
    f"{conda_prefix}/lib:" + os.environ.get('LD_LIBRARY_PATH', '')
)

# 3. Pin to GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.list_physical_devices('GPU')
print("Found GPUs:", gpus)

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
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.list_physical_devices('GPU')
print("GPUS", gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled")
    except RuntimeError as e:
        print(e)
        
# tf.debugging.set_log_device_placement(True)

from utils.preprocess import batch_norm, batch_norm_inverse
from utils.patch_normalization import KerasHybridNormalizer
from utils.visuals import vis_one_var_recon
from utils.metrics import lat_weighted_rmse_one_var

import glob

# %% [markdown]
# ### Data

# %%
def load_patch_data(base_path):
    # Find all timeframe files across all storm directories
    file_paths = glob.glob(os.path.join(base_path, "*", "*.npy"))
    
    data_list = []
    for fp in file_paths:
        patch = np.load(fp)
        data_list.append(patch)
            
    # Stack into a single array (N_samples, 32, 32, 5)
    return np.stack(data_list)

path = '/hdd3/sonia/cyclone/multivar/natlantic'
X_train = load_patch_data(os.path.join(path, 'train'))

X_val = load_patch_data('/hdd3/sonia/cyclone/multivar/satlantic/train')
X_val = np.flip(X_val, axis=1) 
v_idx = 2 # Update this index if v10 is not at index 2!
X_val[..., v_idx] *= -1

X_test = load_patch_data(os.path.join(path, 'test'))

print(X_train.shape, X_val.shape, X_test.shape)

# %% [markdown]
# ### Data normalization

# %%
channel_names = ['slp', 'wind-u', 'wind-v', 'temperature', 'humidity']
normalizer = KerasHybridNormalizer(channel_names)
normalizer.fit(X_train, clamp=True)

X_train_norm = normalizer.normalize(X_train)
X_val_norm = normalizer.normalize(X_val)
X_test_norm = normalizer.normalize(X_test)

print(X_train_norm.shape, X_val_norm.shape, X_test_norm.shape)

# %% [markdown]
# ### Model

# %%
def conv_block(x, filters, strides=1, name=None):
    # use_bias=False is standard when followed by BatchNormalization
    x = layers.Conv2D(filters, (3, 3), padding='same', strides=strides, use_bias=False, name=name)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x) # Swish prevents dead neurons better than ReLU
    return x

def encoder_net(input_shape):
    encoder_inputs = layers.Input(shape=input_shape)

    # ENCODER
    x = conv_block(encoder_inputs, 32, strides=1)
    x = conv_block(x, 128, strides=2) # Down to 16x16
    x = conv_block(x, 256, strides=2) # Down to 8x8

    # BOTTLENECK
    x = conv_block(x, 512, strides=1, name='bottleneck')

    # DECODER
    x = layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = conv_block(x, 128, strides=1)

    x = layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = conv_block(x, 32, strides=1)

    # OUTPUT
    x = layers.Conv2D(5, (3, 3), padding='same', strides=1)(x)
    x = layers.Activation('linear', dtype='float32')(x)
    
    return keras.Model(encoder_inputs, x, name='encoder_net')

# %%
input_shape = (32, 32, 5)
model = encoder_net(input_shape)
model.summary()

# %% [markdown]
# ### Training

# %%
learning_rate = 1e-3
decay_steps = 10000
decay_rate = 0.95

lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
                                                          decay_steps=decay_steps,
                                                          decay_rate=decay_rate
                                                         )

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), 
              loss='mse',
              jit_compile=True)

# model.compile(optimizer='adam',
#     # optimizer=Adam(learning_rate=lr_schedule),
#              loss='mse', 
#             )

es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=20)
mc = ModelCheckpoint('../saved_models/encoder_cnn_patch_multivar.h5', 
                     monitor='val_loss', 
                     mode='min',
                     save_best_only=True,
                    )

BATCH_SIZE = 512

# 1. Clear up raw data to free RAM immediately
del X_train
del X_test
import gc
gc.collect()

# 2. Use a generator to feed the dataset without copying the array
def data_generator(data):
    for sample in data:
        # For an autoencoder, yield the same sample as (input, target)
        yield sample, sample

# 3. Create the dataset from the generator
output_signature = (
    tf.TensorSpec(shape=(32, 32, 5), dtype=tf.float32),
    tf.TensorSpec(shape=(32, 32, 5), dtype=tf.float32)
)

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(X_train_norm),
    output_signature=output_signature
)

# Optimization: Shuffle and Batch
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Repeat for validation
val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(X_val_norm),
    output_signature=output_signature
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Now fit
model.fit(train_dataset, 
          validation_data=val_dataset,
          epochs=100, 
          callbacks=[es, mc])

# %% [markdown]

saved_encoder = keras.models.load_model('../saved_models/encoder_cnn_patch_multivar.h5')
X_recon_norm = saved_encoder.predict(X_test_norm[:5]) # Predict on 5 patches
X_recon = normalizer.denormalize(X_recon_norm) # 5x32x32x5

# %%
import numpy as np
import os
from tqdm import tqdm
from PIL import Image 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# -------- helper to draw black outline & hide ticks --------
def outline_axes(ax, color='k', lw=1.0):
    # ensure frame visible
    ax.set_frame_on(True)
    # black spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)
    # no ticks / labels
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labelleft=False
    )


# %%
fig = plt.figure(figsize=(7, 4), constrained_layout=True)
subfigs = fig.subfigures(1,2, hspace=0.03)

real = normalizer.denormalize(X_test_norm[:5]) # 5x32x32x5
realwm = np.sqrt(real[..., 1]**2 + real[..., 2]**2)
synthwm = np.sqrt(X_recon[..., 1]**2 + X_recon[..., 2]**2)

maxp = synthwm.max()
minp = synthwm.min()
    
# 1. Define discrete levels and colormap
levels = list(range(0,100, 10))
# Start from a discrete viridis
base = plt.cm.get_cmap('viridis', len(levels) - 1)
colors = base(np.arange(base.N))

# Make the first bin (0–10) white
colors[0] = (1.0, 1.0, 1.0, 1.0)  # RGBA white
colors[1] = (0.73, 0.87, 1.00, 1.0) 
colors[2] = (0.50, 0.75, 1.00, 1.0) 
colors[3] = (0.31, 0.63, 0.96, 1.0) 
colors[4] = (0.12, 0.61, 0.80, 1.0) 
colors[5] = (0.00, 0.68, 0.54, 1.0) 
colors[6] = (0.42, 0.75, 0.29, 1.0) 
colors[7] = (0.78, 0.84, 0.17, 1.0) 
colors[8] = (1.00, 0.85, 0.12, 1.0) 

# Build a ListedColormap with modified colors
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)

# real subplot
subfigs[0].suptitle('Real ', fontsize=16, fontweight='bold')
ax0 = subfigs[0].subplots()

i = 0
img = realwm[i] 
print(img.min(), img.max())
img = img- minp 
img = img / (maxp - minp)  # Normalize the image data
img = 100*img
img = img.astype(np.uint8)
print(img.min(), img.max())
ax0.imshow(img, cmap=cmap, norm=norm)
# outline_axes(subfigs[0])
        

# synth subplot
subfigs[1].suptitle('Synthetic ', fontsize=16, fontweight='bold')
ax1 = subfigs[1].subplots()

# plot synth data
img = synthwm[i]
img = img - minp 
img = img / (maxp - minp)  # Normalize the image data
img = 100*img
img = img.astype(np.uint8)
ax1.imshow(img, cmap=cmap, norm=norm)
# outline_axes(subfigs[1])
plt.show()

    # real_levels = np.linspace(0, maxp, len(levels))
    # norm_unnormalized = mcolors.BoundaryNorm(boundaries=real_levels, ncolors=cmap.N)
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_unnormalized)
    # sm.set_array([])  # Dummy array is required for the ScalarMappable to initialize
    # # add_axes defines [left, bottom, width, height] in figure coordinates (0 to 1):
    # cbar_ax = fig.add_axes([0.03, -0.05, 0.94, 0.02]) 
    # cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    # cbar.ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    # cbar.set_label('Wind Magnitude (m/s)', fontsize=12)
# %%
