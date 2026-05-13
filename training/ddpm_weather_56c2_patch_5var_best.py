# %% [markdown]
# ## Setup

# %%
import math
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
# Requires TensorFlow >=2.11 for the GroupNormalization layer.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv3D
from tensorflow.keras.callbacks import *
mixed_precision.set_global_policy('mixed_float16')

# %%
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# %%
tf.__version__

# %% [markdown]
# ## Hyperparameters

# %%
out_fname = sys.argv[-2]
out_name = f'../saved_models/codicast-patch/date/multivar/checkpoints/{out_fname}'
print('will save to', out_name)

batch_size = 256
num_epochs = 400         # Just for the sake of demonstration
total_timesteps = 1000   # 1000
norm_groups = 8          # Number of groups used in GroupNormalization layer
learning_rate = float(sys.argv[-1])

img_size_H = 32
img_size_W = 32
img_channels = 5

first_conv_channels = 64
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [False, False, True, True]
num_res_blocks = 2  # Number of residual blocks

# %% [markdown]
# ## Dataset

# %%
def load_temporal_triplets(base_path):
    past1_list, past2_list, target_list = [], [], []
    
    # Find all storm directories
    storm_dirs = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for storm_dir in storm_dirs:
        # Load all 8 frames for this storm
        frames = []
        for i in range(8):
            filepath = os.path.join(storm_dir, f"{i}.npy")
            if os.path.exists(filepath):
                frames.append(np.load(filepath))
        
        # If the storm has all 8 frames, extract sliding windows of 3
        if len(frames) == 8:
            for t in range(2, 8):
                past1_list.append(frames[t-2]) # T-2
                past2_list.append(frames[t-1]) # T-1
                target_list.append(frames[t])  # T (The diffusion target)
                
    return np.stack(target_list), np.stack(past1_list), np.stack(past2_list)

# Load data 
train_pred, train_past1, train_past2 = \
    load_temporal_triplets('/hdd3/sonia/cyclone/multivar/natlantic/train')
val_pred, val_past1, val_past2 = \
    load_temporal_triplets('/hdd3/sonia/cyclone/multivar/satlantic/train')
test_pred, test_past1, test_past2 = \
    load_temporal_triplets('/hdd3/sonia/cyclone/multivar/natlantic/test')

# Apply Southern Hemisphere V-wind flip if doing it directly in memory
val_pred = np.flip(val_pred, axis=1)
val_past1 = np.flip(val_past1, axis=1)
val_past2 = np.flip(val_past2, axis=1)
v_idx = 2 # Assuming wind-v is index 2
val_pred[..., v_idx] *= -1
val_past1[..., v_idx] *= -1
val_past2[..., v_idx] *= -1

print(train_pred.shape, train_past1.shape, train_past2.shape)

# %%
# train_data_tf = train_data_tf.transpose((0, 2, 3, 1))
# val_data_tf = val_data_tf.transpose((0, 2, 3, 1))
# test_data_tf = test_data_tf.transpose((0, 2, 3, 1))

# print(train_data_tf.shape, val_data_tf.shape, test_data_tf.shape)

# %% [markdown]
# ### Preprocessing
# 
# In terms of preprocessing, we rescale the pixel values in the range `[-1.0, 1.0]`. 
# 
# This is in line with the range of the pixel values that
# was applied by the authors of the [DDPMs paper](https://arxiv.org/abs/2006.11239). 

# %%
from utils.patch_normalization import KerasHybridNormalizer
# from utils.normalization import batch_norm

# %%
channel_names = ['slp', 'wind-u', 'wind-v', 'temperature', 'humidity']
normalizer = KerasHybridNormalizer(channel_names)
normalizer.fit(train_pred, clamp=False)

train_norm_pred = normalizer.normalize(train_pred)
train_norm_past1 = normalizer.normalize(train_past1)
train_norm_past2 = normalizer.normalize(train_past2)

val_norm_pred = normalizer.normalize(val_pred)
val_norm_past1 = normalizer.normalize(val_past1)
val_norm_past2 = normalizer.normalize(val_past2)

test_norm_pred = normalizer.normalize(test_pred)
test_norm_past1 = normalizer.normalize(test_past1)
test_norm_past2 = normalizer.normalize(test_past2)


print(train_norm_pred.shape, train_norm_past1.shape, train_norm_past2.shape)
print(val_norm_pred.shape, val_norm_past1.shape, val_norm_past2.shape)
print(test_norm_pred.shape, test_norm_past1.shape, test_norm_past2.shape)

# %% [markdown]
# ## Gaussian diffusion utilities
# 
# We define the **forward process** and the **reverse process** as a separate utility. Most of the code in this utility has been borrowed
# from the original implementation with some slight modifications.

# %%
from layers.diffusion import GaussianDiffusion

# %% [markdown]
# ## Network architecture
# 
# U-Net, originally developed for semantic segmentation, is an architecture that is
# widely used for implementing diffusion models but with some slight modifications:
# 
# 1. The network accepts two inputs: Image and time step
# 2. Self-attention between the convolution blocks once we reach a specific resolution
# (16x16 in the paper)
# 3. Group Normalization instead of weight normalization
# 
# We implement most of the things as used in the original paper. We use the
# `swish` activation function throughout the network. We use the variance scaling
# kernel initializer.
# 
# The only difference here is the number of groups used for the
# `GroupNormalization` layer. For the flowers dataset,
# we found that a value of `groups=8` produces better results
# compared to the default value of `groups=32`. Dropout is optional and should be
# used where chances of over fitting is high. In the paper, the authors used dropout
# only when training on CIFAR10.

# %%
from tensorflow.keras.models import load_model

pretrained_encoder_full = load_model('../saved_models/debug_encoder_cnn_patch_multivar.h5')
pretrained_encoder_full.summary()

# %%
bottleneck_layer = pretrained_encoder_full.get_layer('bottleneck').output
pretrained_encoder = tf.keras.Model(inputs=pretrained_encoder_full.input, 
                                    outputs=bottleneck_layer, 
                                    name='encoder')
# # Extract the first 5 layers
# first_five_layers = pretrained_encoder.layers[:5]
# # Display the first four layers to confirm
# for i, layer in enumerate(first_five_layers):
#     print(f"Layer {i}: {layer}")

# # Create a new model using these layers
# # Get the input of the pre-trained model
# input_layer = pretrained_encoder.input

# # Get the output of the fourth layer
# output_layer = first_five_layers[-1].output

# # Create the new model
# pretrained_encoder = tf.keras.Model(inputs=input_layer, outputs=output_layer)

# Print the summary of the new model
pretrained_encoder.summary()

# %%
for layer in pretrained_encoder.layers:
    layer.trainable = False

pretrained_encoder._name = 'encoder'

# %%
from layers.denoiser import build_unet_model_c2

# %%
# Build the unet model
network = build_unet_model_c2(
    img_size_H=img_size_H,
    img_size_W=img_size_W,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    first_conv_channels=first_conv_channels,
    activation_fn=keras.activations.swish,
    encoder=pretrained_encoder,
)

# %%
# network.summary()

# %% [markdown]
# ## Training
# 
# We follow the same setup for training the diffusion model as described
# in the paper. We use `Adam` optimizer with a learning rate of `2e-4`.
# We use `EMA` (Exponential Moving Average) on model parameters with a decay factor of 0.999. We
# treat our model as noise prediction network i.e. at every training step, we
# input a batch of images and corresponding time steps to our UNet,
# and the network outputs the noise as predictions.
# 
# The only difference is that we aren't using the Kernel Inception Distance (KID)
# or Frechet Inception Distance (FID) for evaluating the quality of generated
# samples during training. This is because both these metrics are compute heavy
# and are skipped for the brevity of implementation.
# 
# **Note: ** We are using mean squared error as the loss function which is aligned with
# the paper, and theoretically makes sense. In practice, though, it is also common to
# use mean absolute error or Huber loss as the loss function.

# %%
class DiffusionModel(keras.Model):
    def __init__(self, network, ema_network, timesteps, gdf_util, ema=0.999):
        super().__init__()
        self.network = network  # denoiser or noise predictor
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.ema = ema

    def train_step(self, data):
        # Unpack the data
        (images, image_input_past1, image_input_past2), y = data
        
        images = tf.cast(images, tf.float32)
        image_input_past1 = tf.cast(image_input_past1, tf.float32)
        image_input_past2 = tf.cast(image_input_past2, tf.float32)
        
        # 1. Get the batch size
        batch_size = tf.shape(images)[0]
        
        # 2. Sample timesteps uniformly
        t = tf.random.uniform(minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64)

        with tf.GradientTape() as tape:
            # 3. Sample random noise to be added to the images in the batch
            noise = tf.random.normal(shape=tf.shape(images), dtype=tf.float32)
            # print("noise.shape:", noise.shape)
            
            # 4. Diffuse the images with noise
            images_t = self.gdf_util.q_sample(images, t, noise)
            # print("images_t.shape:", images_t.shape)
            
            # 5. Pass the diffused images and time steps to the network
            pred_noise = self.network([images_t, t, image_input_past1, image_input_past2], training=True)
            # print("pred_noise.shape:", pred_noise.shape)
            
            # 6. Calculate the loss
            loss = self.loss(noise, pred_noise)
            
            # Scale the loss to prevent float16 underflow
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        # 7. Get the gradients
        scaled_gradients = tape.gradient(scaled_loss, self.network.trainable_weights)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients) # unscale

        # 8. Update the weights of the network
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # 9. Updates the weight values for the network with EMA weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # 10. Return loss values
        return {"loss": loss}

    
    def test_step(self, data):
        # Unpack the data
        (images, image_input_past1, image_input_past2), y = data
        
        images = tf.cast(images, tf.float32)
        image_input_past1 = tf.cast(image_input_past1, tf.float32)
        image_input_past2 = tf.cast(image_input_past2, tf.float32)

        # 1. Get the batch size
        batch_size = tf.shape(images)[0]
        
        # 2. Sample timesteps uniformly
        t = tf.random.uniform(minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64)

        # 3. Sample random noise to be added to the images in the batch
        noise = tf.random.normal(shape=tf.shape(images), dtype=tf.float32)
        
        # 4. Diffuse the images with noise
        images_t = self.gdf_util.q_sample(images, t, noise)
        
        # 5. Pass the diffused images and time steps to the network
        pred_noise = self.network([images_t, t, image_input_past1, image_input_past2], training=False)
        
        # 6. Calculate the loss
        loss = self.loss(noise, pred_noise)

        # 7. Return loss values
        return {"loss": loss}



# Build the unet model
network = build_unet_model_c2(
    img_size_H=img_size_H,
    img_size_W=img_size_W,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    first_conv_channels=first_conv_channels,
    activation_fn=keras.activations.swish,
    encoder=pretrained_encoder,
    interpolation="bilinear"
)

ema_network = build_unet_model_c2(
    img_size_H=img_size_H,
    img_size_W=img_size_W,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    first_conv_channels=first_conv_channels,
    activation_fn=keras.activations.swish,
    encoder=pretrained_encoder,
    interpolation="bilinear"
)
ema_network.set_weights(network.get_weights())  # Initially the weights are the same

# %%
# ema_network.summary()

# %% [markdown]
# ### Training

# %%
train_dataset = tf.data.Dataset.from_tensor_slices(
    ((train_norm_pred, train_norm_past1, train_norm_past2), train_norm_pred)
)
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices(
    ((val_norm_pred, val_norm_past1, val_norm_past2), val_norm_pred)
).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# %%
# from loss.loss import lat_weighted_loss_mse_56deg

# %%
learning_rate = 1e-4
decay_steps = 10000
decay_rate = 0.95


# Get an instance of the Gaussian Diffusion utilities
gdf_util = GaussianDiffusion(timesteps=total_timesteps)

# Get the model
model = DiffusionModel(
    network=network,
    ema_network=ema_network,
    gdf_util=gdf_util,
    timesteps=total_timesteps,
)

# %%
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
                                                          decay_steps=decay_steps,
                                                          decay_rate=decay_rate
                                                         )


# Calculate steps
train_steps = len(train_dataset)
val_steps = len(val_dataset)

print(f"Total training samples: {len(train_norm_past1)}")
print(f"Steps per epoch: {train_steps}")
print(f"Validation steps: {val_steps}")

cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=out_name, 
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min',  
    save_freq='epoch',
    verbose=1,
)

# 2. Define the Early Stopping Callback
early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',        # The metric to watch (Keras adds 'val_' to the loss returned by test_step)
    patience=50,               # Number of epochs to wait for improvement before stopping
    restore_best_weights=True, # Automatically restores the weights from the epoch with the best val_loss
    verbose=1                  # Prints a message when early stopping is triggered
)

# Compile the model
model.compile(
              loss=keras.losses.MeanSquaredError(),
              optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
              jit_compile=True
             )

# Train the model
model.fit(train_dataset,
          validation_data=val_dataset,
          epochs=num_epochs,
          callbacks=[cp_callback, early_stopping_callback]
         )

# %%
# Save weights

# %%
# Restore weights
# model.load_weights('../checkpoints/ddpm_weather_56c2_56_multivar_cp3')

# %%

