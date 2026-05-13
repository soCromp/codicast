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
output_folder = f'/mnt/data/sonia/codicast-patch/date/multivar/out-{name}'
network_path = f'/mnt/data/sonia/codicast-patch/date/multivar/checkpoints/{name}'

batch_size = 1024
total_timesteps = 1000
norm_groups = 8          # Number of groups used in GroupNormalization layer

img_size_H = 32
img_size_W = 32
img_channels = 5

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
from layers.denoiser import build_unet_model_c2 

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
        # Unpack the data
        (images, image_input_past1, image_input_past2), y = data
        
        # 1. Get the batch size
        batch_size = tf.shape(images)[0]
        
        # 2. Sample timesteps uniformly
        t = tf.random.uniform(minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64)

        with tf.GradientTape() as tape:
            # 3. Sample random noise to be added to the images in the batch
            noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
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

        # 1. Get the batch size
        batch_size = tf.shape(images)[0]
        
        # 2. Sample timesteps uniformly
        t = tf.random.uniform(minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64)

        # 3. Sample random noise to be added to the images in the batch
        noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
        
        # 4. Diffuse the images with noise
        images_t = self.gdf_util.q_sample(images, t, noise)
        
        # 5. Pass the diffused images and time steps to the network
        pred_noise = self.network([images_t, t, image_input_past1, image_input_past2], training=False)
        
        # 6. Calculate the loss
        loss = self.loss(noise, pred_noise)

        # 7. Return loss values
        return {"loss": loss}
    
# %%
# 1. Build the unet models (Do not load weights here!)
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
def load_temporal_triplets(base_path):
    past1_list, past2_list, target_list = [], [], []
    
    # Find all storm directories
    storm_dirs = sorted([os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    
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

# Load data (Update paths as needed)
train_pred, train_past1, train_past2 = \
    load_temporal_triplets('/home/cyclone/train/multivar/0.25/date/natlantic/train')
val_pred, val_past1, val_past2 = \
    load_temporal_triplets('/home/cyclone/train/multivar/0.25/date/satlantic/val')
test_pred, test_past1, test_past2 = \
    load_temporal_triplets('/home/cyclone/train/multivar/0.25/date/natlantic/test')
    
sids = sids([d for d in os.listdir('/home/cyclone/train/multivar/0.25/date/natlantic/test') \
    if os.path.isdir(os.path.join('/home/cyclone/train/multivar/0.25/date/natlantic/test', d))])

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

from utils.patch_normalization import KerasHybridNormalizer

channel_names = ['slp', 'wind-u', 'wind-v', 'temperature', 'humidity']
normalizer = KerasHybridNormalizer(channel_names)
normalizer.fit(train_pred, clamp=True)

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

def generate_images(model, original_samples, original_samples_past1, original_samples_past2, frame_num=None):
    """
    @model: trained denoiser
    @original_samples: it just provides the shape, does not involve generation
    @original_samples_past: conditions from the past
    """
    num_images = original_samples.shape[0]
    img_size_H = original_samples.shape[1]
    img_size_W = original_samples.shape[2]
    img_channels = original_samples.shape[3]
    total_timesteps = model.timesteps  # Ensure this is defined in your model

    # 1. Randomly sample noise (starting point for reverse process)
    samples = tf.random.normal(shape=(num_images, img_size_H, img_size_W, img_channels), dtype=tf.float32)
    
    desc = f"Denoising Frame {frame_num}" if frame_num is not None else "Denoising"
    
    # 2. Sample from the model iteratively
    for t in tqdm(reversed(range(0, total_timesteps)), total=total_timesteps, desc=desc, leave=False):
        tt = tf.cast(tf.fill([num_images], t), dtype=tf.int64)
        # pred_noise = model.ema_network([samples, tt, original_samples_past1, original_samples_past2],
        #                                        training=False)
        pred_noise = fast_predict_noise(model.ema_network, samples, tt, original_samples_past1, original_samples_past2)
        samples = model.gdf_util.p_sample(pred_noise, samples, tt, clip_denoised=True)
        
    # 3. Return generated samples and original samples
    return original_samples, samples

def predict_autoregressive(model, initial_inputs, prediction_horizon):
    predictions = []
    
    # These now have shape (batch_size, H, W, C)
    original_sample, sample_past1, sample_past2 = initial_inputs[0], initial_inputs[1], initial_inputs[2]  

    for step in range(prediction_horizon):
        # Generate the next step for the ENTIRE batch at once
        original_sample, generated_sample = generate_images(model, original_sample, sample_past1, sample_past2, frame_num=step+1)
        
        predictions.append(generated_sample)

        # Shift the autoregressive window
        sample_past1 = sample_past2
        sample_past2 = generated_sample
        
    # Stack along axis=1 to create a distinct time dimension.
    # Resulting shape: (batch_size, prediction_horizon, H, W, C)
    predictions = tf.stack(predictions, axis=1) 
    return predictions.numpy()

# %%
# channels = ['geopotential_500', 'temperature_850', 
#             '2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind']
channels = ['sea_level_pressure', 
            'u_component_of_wind_500', 'v_component_of_wind_500', 'temperature_925', 'humidity_500']

num_sample = 899  # 2918 for entire default test set; 899 for my date split
num_channel = 5
num_lead = 8


rmse_matrix = np.zeros((num_sample, num_channel, num_lead))
acc_matrix = np.zeros((num_sample, num_channel, num_lead))
    
    
inference_batch_size = 512

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
    # Returns shape: (current_batch, num_lead, H, W, C)
    future_predictions = predict_autoregressive(model, initial_inputs, prediction_horizon=num_lead)

    # 3. Unnormalize
    generated_samples_unnormlalized = normalizer.denormalize(future_predictions)
    
    # 4. Save to disk
    for b in range(current_batch):
        # Calculate the actual sample index for saving
        sample_idx = z + b 
        
        os.makedirs(os.path.join(output_folder, sids[sample_idx]), exist_ok=True)
        for t in range(num_lead):
            np.save(
                os.path.join(output_folder, sids[sample_idx], f'{t}.npy'), 
                generated_samples_unnormlalized[b, t]
            )
            