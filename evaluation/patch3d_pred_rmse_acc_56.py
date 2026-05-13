# %% [markdown]
# ## Setup

# %%
import math
import numpy as np
import matplotlib.pyplot as plt
import glob

# Requires TensorFlow >=2.11 for the GroupNormalization layer.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv3D
from tensorflow.keras.callbacks import *
mixed_precision.set_global_policy('mixed_float16')

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm

# # Enable memory growth to stop TF from hoarding VRAM
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# %% [markdown]
# ## Hyperparameters

# %%
name = sys.argv[-1]
output_folder = f'../out-{name}'
network_path = f'/hdd3/sonia/codicast/saved_models/3d_checkpoints/{name}'

batch_size = 1024
total_timesteps = 1000
norm_groups = 8          # Number of groups used in GroupNormalization layer

img_size_H = 32
img_size_W = 32
img_channels = 5
seq_len = 6

first_conv_channels = 64
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [False, False, True, True]
num_res_blocks = 2  # Number of residual blocks

# %%
from layers.diffusion import GaussianDiffusion
from tensorflow.keras.models import load_model

encoder = keras.models.load_model('../saved_models/encoder_cnn_patch_multivar.h5')
encoder.trainable = False
bottleneck_layer = encoder.get_layer('bottleneck').output
pretrained_encoder = tf.keras.Model(inputs=encoder.input, outputs=bottleneck_layer)
for layer in pretrained_encoder.layers:
    layer.trainable = False

pretrained_encoder._name = 'encoder'

# %%
from layers.denoiser import build_unet_model_c2_3d

# %% [markdown]
# ## Model loading

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
        (images, image_input_past1, image_input_past2), y = data
        
        # Dynamically get shapes
        shape = tf.shape(images)
        B, T, H, W, C = shape[0], shape[1], shape[2], shape[3], shape[4]
        
        # Sample timesteps uniformly (One timestep per video)
        t = tf.random.uniform(minval=0, maxval=self.timesteps, shape=(B,), dtype=tf.int64)

        with tf.GradientTape() as tape:
            # Sample random noise 
            noise = tf.random.normal(shape=shape, dtype=images.dtype)
            
            # --- THE 4D DIFFUSION FOLD ---
            images_folded = tf.reshape(images, [-1, H, W, C])
            noise_folded = tf.reshape(noise, [-1, H, W, C])
            t_folded = tf.repeat(t, repeats=T, axis=0)
            
            # Diffuse the folded 4D images
            images_t_folded = self.gdf_util.q_sample(images_folded, t_folded, noise_folded)
            
            # Unfold back to 5D video
            images_t = tf.reshape(images_t_folded, [-1, T, H, W, C])
            # -----------------------------
            
            # Pass the diffused images and time steps to the network
            pred_noise = self.network([images_t, t, image_input_past1, image_input_past2], training=True)
            
            # Calculate the loss
            loss = self.loss(noise, pred_noise)
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        # Get the gradients and apply
        scaled_gradients = tape.gradient(scaled_loss, self.network.trainable_weights)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # Update EMA weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        return {"loss": loss}

    
    def test_step(self, data):
        (images, image_input_past1, image_input_past2), y = data

        shape = tf.shape(images)
        B, T, H, W, C = shape[0], shape[1], shape[2], shape[3], shape[4]
        
        t = tf.random.uniform(minval=0, maxval=self.timesteps, shape=(B,), dtype=tf.int64)
        noise = tf.random.normal(shape=shape, dtype=images.dtype)
        
        # --- THE 4D DIFFUSION FOLD ---
        images_folded = tf.reshape(images, [-1, H, W, C])
        noise_folded = tf.reshape(noise, [-1, H, W, C])
        t_folded = tf.repeat(t, repeats=T, axis=0)
        
        images_t_folded = self.gdf_util.q_sample(images_folded, t_folded, noise_folded)
        images_t = tf.reshape(images_t_folded, [-1, T, H, W, C])
        # -----------------------------
        
        pred_noise = self.network([images_t, t, image_input_past1, image_input_past2], training=False)
        loss = self.loss(noise, pred_noise)

        return {"loss": loss}
    
# %%
# 1. Build the unet models (Do not load weights here!)
network = build_unet_model_c2_3d(
    seq_len=seq_len,
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

ema_network = build_unet_model_c2_3d(
    seq_len=seq_len,
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

# 2. Get an instance of the Gaussian Diffusion utilities
gdf_util = GaussianDiffusion(timesteps=total_timesteps)

# 3. Assemble the DiffusionModel Wrapper
model = DiffusionModel(
    network=network,
    ema_network=ema_network,
    gdf_util=gdf_util,
    timesteps=total_timesteps,
)

# 4. NOW load the weights into the top-level model
# .expect_partial() tells Keras not to warn us about missing optimizer weights, 
# since we aren't compiling the model for training here.
model.load_weights(network_path).expect_partial()


# %% [markdown]
# ## Results

# %%
from utils.normalization import batch_norm, batch_norm_reverse
from utils.metrics import lat_weighted_rmse_one_var, lat_weighted_acc_one_var

# %%
def load_temporal_triplets_video(base_path, seq_len=8):
    past1_list, past2_list, target_list = [], [], []
    
    storm_dirs = sorted([os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    # print(storm_dirs)
    
    for storm_dir in storm_dirs:
        frames = []
        # Let's assume some storms might have more than 8 frames; load them all
        num_files = len(glob.glob(os.path.join(storm_dir, "*.npy")))
        # print(num_files)
        for i in range(num_files):
            filepath = os.path.join(storm_dir, f"{i}.npy")
            if os.path.exists(filepath):
                frames.append(np.load(filepath))
        
        # We need a sliding window that yields: Past2, Past1, and then the next 8 frames
        # Total needed window size = 2 (past) + 8 (future) = 10
        if len(frames) >= (2 + seq_len):
            for t in range(2, len(frames) - seq_len + 1):
                past1_list.append(frames[t-2])     # T-2
                past2_list.append(frames[t-1])     # T-1
                
                # The target is now a sequence of 8 frames!
                target_video = np.stack(frames[t : t+seq_len], axis=0)
                target_list.append(target_video)
                
    return np.stack(target_list), np.stack(past1_list), np.stack(past2_list)


train_pred, train_past1, train_past2 = \
    load_temporal_triplets_video('/hdd3/sonia/cyclone/multivar/natlantic/train', seq_len=seq_len)
test_pred, test_past1, test_past2 = \
    load_temporal_triplets_video('/hdd3/sonia/cyclone/multivar/natlantic/test', seq_len=seq_len)

print(train_pred.shape, train_past1.shape, train_past2.shape)
print(test_pred.shape, test_past1.shape, test_past2.shape)

sids = sorted([d for d in os.listdir('/hdd3/sonia/cyclone/multivar/natlantic/test') \
    if os.path.isdir(os.path.join('/hdd3/sonia/cyclone/multivar/natlantic/test', d))])

# %%

from utils.patch_normalization import KerasHybridNormalizer

channel_names = ['slp', 'wind-u', 'wind-v', 'temperature', 'humidity']
normalizer = KerasHybridNormalizer(channel_names)
normalizer.fit(train_pred, clamp=True)

# train_norm_pred = normalizer.normalize(train_pred)
# train_norm_past1 = normalizer.normalize(train_past1)
# train_norm_past2 = normalizer.normalize(train_past2)

test_norm_pred = normalizer.normalize(test_pred)
test_norm_past1 = normalizer.normalize(test_past1)
test_norm_past2 = normalizer.normalize(test_past2)

print(test_norm_pred.shape, test_norm_past1.shape, test_norm_past2.shape)

# test_dataset = tf.data.Dataset.from_tensor_slices(
#     ((test_norm_pred, test_norm_past1, test_norm_past2), test_norm_pred)
# ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# %% [markdown]
# ### RSME & ACC

# %%
from utils.normalization import batch_norm, batch_norm_reverse
from utils.metrics import lat_weighted_rmse_one_var, lat_weighted_acc_one_var

# %%
import tensorflow as tf
import numpy as np


# Force XLA compilation for the inference forward pass
@tf.function(jit_compile=True)
def fast_predict_noise(model_ema, samples, tt, past1, past2):
    return model_ema([samples, tt, past1, past2], training=False)

def generate_images(model, original_samples, original_samples_past1, original_samples_past2):
    # 1. Properly unpack the 5D shape
    shape = original_samples.shape
    num_images, seq_len, img_size_H, img_size_W, img_channels = shape[0], shape[1], shape[2], shape[3], shape[4]
    total_timesteps = model.timesteps 

    # 2. Sample 5D noise
    samples = tf.random.normal(shape=shape, dtype=tf.float32)
    
    desc = "Denoising"
    
    for t in tqdm(reversed(range(0, total_timesteps)), total=total_timesteps, desc=desc, leave=False):
        tt = tf.cast(tf.fill([num_images], t), dtype=tf.int64)
        
        # Predict 5D noise
        pred_noise = fast_predict_noise(model.ema_network, samples, tt, original_samples_past1, original_samples_past2)
        
        # --- THE 4D DIFFUSION FOLD FOR P_SAMPLE ---
        samples_folded = tf.reshape(samples, [-1, img_size_H, img_size_W, img_channels])
        pred_noise_folded = tf.reshape(pred_noise, [-1, img_size_H, img_size_W, img_channels])
        tt_folded = tf.repeat(tt, repeats=seq_len, axis=0)
        
        # Apply p_sample on the folded 4D tensors
        samples_folded = model.gdf_util.p_sample(pred_noise_folded, samples_folded, tt_folded, clip_denoised=True)
        
        # Unfold back to 5D video for the next U-Net pass
        samples = tf.reshape(samples_folded, [-1, seq_len, img_size_H, img_size_W, img_channels])
        # ------------------------------------------
        
    return original_samples, samples

# def predict_autoregressive(model, initial_inputs, prediction_horizon):
#     predictions = []
    
#     # These now have shape (batch_size, H, W, C)
#     original_sample, sample_past1, sample_past2 = initial_inputs[0], initial_inputs[1], initial_inputs[2]  

#     for step in range(prediction_horizon):
#         # Generate the next step for the ENTIRE batch at once
#         original_sample, generated_sample = generate_images(model, original_sample, sample_past1, sample_past2, frame_num=step+1)
        
#         predictions.append(generated_sample)

#         # Shift the autoregressive window
#         sample_past1 = sample_past2
#         sample_past2 = generated_sample
        
#     # Stack along axis=1 to create a distinct time dimension.
#     # Resulting shape: (batch_size, prediction_horizon, H, W, C)
#     predictions = tf.stack(predictions, axis=1) 
#     return predictions.numpy()

# %%
# channels = ['geopotential_500', 'temperature_850', 
#             '2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind']
channels = ['sea_level_pressure', 
            'u_component_of_wind_500', 'v_component_of_wind_500', 'temperature_925', 'humidity_500']

num_sample = 899  # 2918 for entire default test set; 899 for my date split
num_channel = 5


rmse_matrix = np.zeros((num_sample, num_channel, seq_len))
acc_matrix = np.zeros((num_sample, num_channel, seq_len))
    
    
inference_batch_size = 32

for z in tqdm(range(0, num_sample, inference_batch_size)):
    # Calculate the current batch size (this prevents errors on the final, smaller batch)
    current_batch = min(inference_batch_size, num_sample - z)
    
    # 1. Slice starting conditions for the entire batch
    # This grabs t=0 for z, z+1, ..., z+current_batch-1
    # Shape becomes (current_batch, H, W, C)
    batch_pred_t0 = tf.convert_to_tensor(test_norm_pred[z : z + current_batch], dtype=tf.float32)
    batch_past1_t0 = tf.convert_to_tensor(test_norm_past1[z : z + current_batch], dtype=tf.float32)
    batch_past2_t0 = tf.convert_to_tensor(test_norm_past2[z : z + current_batch], dtype=tf.float32)
    
    initial_inputs = [batch_pred_t0, batch_past1_t0, batch_past2_t0]

    # 2. Predict the future for the entire batch
    # Returns shape: (current_batch, seq_len, H, W, C)
    _, generated_sample = generate_images(model, batch_pred_t0, batch_past1_t0, batch_past2_t0)

    # 3. Unnormalize
    generated_samples_unnormlalized = normalizer.denormalize(generated_sample.numpy())
    
    # 4. Save to disk
    for b in range(current_batch):
        # Calculate the actual sample index for saving
        sample_idx = z + b 
        
        os.makedirs(os.path.join(output_folder, sids[sample_idx]), exist_ok=True)
        for t in range(seq_len):
            np.save(
                os.path.join(output_folder, sids[sample_idx], f'{t}.npy'), 
                generated_samples_unnormlalized[b, t]
            )
            