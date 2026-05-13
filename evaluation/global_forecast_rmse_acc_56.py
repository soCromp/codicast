# %% [markdown]
# ## Setup

# %%
import math
import numpy as np
import matplotlib.pyplot as plt

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
batch_size = 1024
num_epochs = 800         # Just for the sake of demonstration
total_timesteps = 750   # 1000
norm_groups = 8          # Number of groups used in GroupNormalization layer
learning_rate = 1e-4

img_size_H = 32
img_size_W = 64
img_channels = 5

first_conv_channels = 64
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [False, False, True, True]
num_res_blocks = 2  # Number of residual blocks

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

pretrained_encoder = load_model('../saved_models/encoder_cnn_56deg_multivar.h5')
pretrained_encoder.summary()

# %%
# Extract the first 5 layers
first_five_layers = pretrained_encoder.layers[:5]

# Display the first four layers to confirm
for i, layer in enumerate(first_five_layers):
    print(f"Layer {i}: {layer}")

# Create a new model using these layers
# Get the input of the pre-trained model
input_layer = pretrained_encoder.input

# Get the output of the fourth layer
output_layer = first_five_layers[-1].output

# Create the new model
pretrained_encoder = tf.keras.Model(inputs=input_layer, outputs=output_layer)

# Print the summary of the new model
pretrained_encoder.summary()

# %%
for layer in pretrained_encoder.layers:
    layer.trainable = False

pretrained_encoder._name = 'encoder'

# %%
from layers.denoiser import build_unet_model_c2_orig#, build_unet_model_c2_no_cross_attn, build_unet_model_c2_no_encoder, build_unet_model_c2_no_cross_attn_encoder

# %%
# Build the unet model
network = build_unet_model_c2_orig(
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
            print("noise.shape:", noise.shape)
            
            # 4. Diffuse the images with noise
            images_t = self.gdf_util.q_sample(images, t, noise)
            print("images_t.shape:", images_t.shape)
            
            # 5. Pass the diffused images and time steps to the network
            pred_noise = self.network([images_t, t, image_input_past1, image_input_past2], training=True)
            print("pred_noise.shape:", pred_noise.shape)
            
            # 6. Calculate the loss
            loss = self.loss(noise, pred_noise)

        # 7. Get the gradients
        gradients = tape.gradient(loss, self.network.trainable_weights)

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

# %% [markdown]
# ### Load trained model

# %%
# Build the unet model
network = build_unet_model_c2_orig(
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

ema_network = build_unet_model_c2_orig(
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
ema_network.set_weights(network.get_weights())  # Initially the weights are the same

# %%
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
## -------------------- ablation study --------------------
model.load_weights('/mnt/data/sonia/codicast-out/date/multivar/checkpoints/ddpm_onepast_56c2_56_multivar_cp3')  # CoDiCast with 1000 steps
# model.load_weights('../checkpoints/ddpm_weather_56c2_56_5var_cp3_no_cross_attn')
# model.load_weights('../checkpoints/ddpm_weather_56c2_56_5var_cp3_no_encoder')
# model.load_weights('../checkpoints/ddpm_weather_56c2_56_5var_cp3_no_cross_attn_encoder')

## -------------------- diffusion steps --------------------
# model.load_weights('../checkpoints/ddpm_weather_56c2_56_5var_cp3_750')

# %% [markdown]
# ## Results

# %%
from utils.normalization import batch_norm, batch_norm_reverse
from utils.metrics import lat_weighted_rmse_one_var, lat_weighted_acc_one_var

# %%
resolution_folder = '56degree'
resolution = '5.625'  
var_num = '5'

output_folder = '/mnt/data/sonia/codicast-out/date/multivar/raw-out-onepast/test'

# test_data_tf = np.load("/mnt/data/sonia/codicast-data/default/concat_2017_2018_" + resolution + "_" + var_num + "var.npy")
test_data_tf = np.load("/mnt/data/sonia/codicast-data/multivar/concat_2016_2024_" + resolution + "_" + var_num + "var.npy")
test_data_tf = test_data_tf.transpose((0, 2, 3, 1))
test_data_tf.shape

# %%
test_data_tf_norm = batch_norm(test_data_tf, test_data_tf.shape, batch_size=1460)
test_data_tf_norm.shape

# %%
test_data_tf_norm_pred = test_data_tf_norm[1:]
test_data_tf_norm_past1 = test_data_tf_norm[:-1]
test_data_tf_norm_past2 = test_data_tf_norm[:-1]

print(test_data_tf_norm_pred.shape, test_data_tf_norm_past1.shape, test_data_tf_norm_past2.shape)

# %% [markdown]
# ### RSME & ACC

# %%
from utils.normalization import batch_norm, batch_norm_reverse
from utils.metrics import lat_weighted_rmse_one_var, lat_weighted_acc_one_var

# %%
import tensorflow as tf
import numpy as np

def generate_images(model, original_samples, original_samples_past1, original_samples_past2):
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
    
    # 2. Sample from the model iteratively
    for t in reversed(range(0, total_timesteps)):
        tt = tf.cast(tf.fill([num_images], t), dtype=tf.int64)
        pred_noise = model.ema_network([samples, tt, original_samples_past1, original_samples_past2],
                                               training=False
                                              )
        samples = model.gdf_util.p_sample(pred_noise, samples, tt, clip_denoised=True)
        
    # 3. Return generated samples and original samples
    return original_samples, samples
    # return original_samples.numpy(), samples.numpy()

# %%
# def predict_autoregressive(model, initial_inputs, prediction_horizon):
    
#     predictions = []
    
#     original_sample, sample_past1, sample_past2 = initial_inputs[0], initial_inputs[1], initial_inputs[2]  # t, t-2, t-1

#     for _ in range(prediction_horizon):
#         # Predict the next time step
#         original_sample, generated_sample = generate_images(model, original_sample, sample_past1, sample_past2)
        
#         # print("original_sample.shape:", original_sample.shape, "generated_sample.shape:", generated_sample.shape)
        
#         # Append the prediction to the list of predictions
#         predictions.append(generated_sample)

#         sample_past1 = sample_past2
#         sample_past2 = generated_sample
        

#     # Concatenate predictions along the time steps axis
#     predictions = np.concatenate(predictions, axis=0)
#     return predictions

def predict_autoregressive(model, initial_inputs, prediction_horizon):
    predictions = []
    
    # These now have shape (batch_size, H, W, C)
    original_sample, sample_past1, sample_past2 = initial_inputs[0], initial_inputs[1], initial_inputs[2]  

    for _ in range(prediction_horizon):
        # Generate the next step for the ENTIRE batch at once
        original_sample, generated_sample = generate_images(model, original_sample, sample_past1, sample_past2)
        
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


# for z in tqdm(range(num_sample)):
#     #print("sample #", z)
#     original_samples = tf.convert_to_tensor(test_data_tf_norm_pred[z:z+num_lead], dtype=tf.float32)
#     original_samples_past1 = tf.convert_to_tensor(test_data_tf_norm_past1[z:z+num_lead], dtype=tf.float32)
#     original_samples_past2 = tf.convert_to_tensor(test_data_tf_norm_past2[z:z+num_lead], dtype=tf.float32)
    
#     # print(original_samples.shape, original_samples_past1.shape, original_samples_past2.shape)
    
#     initial_inputs = [tf.convert_to_tensor(original_samples[0:1], dtype=tf.float32),
#                   tf.convert_to_tensor(original_samples_past1[0:1], dtype=tf.float32), 
#                   tf.convert_to_tensor(original_samples_past2[0:1], dtype=tf.float32)
#                  ]

#     future_predictions = predict_autoregressive(model, initial_inputs, prediction_horizon=num_lead)

#     original_samples_unnormlalized = batch_norm_reverse(test_data_tf, test_data_tf.shape, 1459, original_samples)
#     generated_samples_unnormlalized = batch_norm_reverse(test_data_tf, test_data_tf.shape, 1459, future_predictions)
    
#     # print(original_samples_unnormlalized.shape, generated_samples_unnormlalized.shape)
    
#     os.makedirs(os.path.join(output_folder, str(z)), exist_ok=True)
#     for t in range(num_lead):
#         np.save(os.path.join(output_folder, str(z), f'{t}.npy'), generated_samples_unnormlalized[t])
    
    
inference_batch_size = 512

for z in tqdm(range(0, num_sample, inference_batch_size)):
    # Calculate the current batch size (this prevents errors on the final, smaller batch)
    current_batch = min(inference_batch_size, num_sample - z)
    
    # 1. Slice starting conditions for the entire batch
    # This grabs t=0 for z, z+1, ..., z+current_batch-1
    # Shape becomes (current_batch, H, W, C)
    batch_pred_t0 = tf.convert_to_tensor(test_data_tf_norm_pred[z : z + current_batch], dtype=tf.float32)
    batch_past1_t0 = tf.convert_to_tensor(test_data_tf_norm_past1[z : z + current_batch], dtype=tf.float32)
    batch_past2_t0 = tf.convert_to_tensor(test_data_tf_norm_past2[z : z + current_batch], dtype=tf.float32)
    
    initial_inputs = [batch_pred_t0, batch_past1_t0, batch_past2_t0]

    # 2. Predict the future for the entire batch
    # Returns shape: (current_batch, num_lead, H, W, C)
    future_predictions = predict_autoregressive(model, initial_inputs, prediction_horizon=num_lead)

    # 3. Unnormalize
    # Depending on how batch_norm_reverse is written, it might not handle 5D tensors well.
    # To be safe, we flatten the batch and time dimensions into one, unnormalize, then reshape back.
    _, H, W, C = future_predictions.shape[1:]
    flat_preds = tf.reshape(future_predictions, (current_batch * num_lead, H, W, C))
    
    unnorm_flat = batch_norm_reverse(test_data_tf, test_data_tf.shape, 1459, flat_preds)
    
    # Reshape back to (batch, time, H, W, C)
    generated_samples_unnormlalized = tf.reshape(unnorm_flat, (current_batch, num_lead, H, W, C)).numpy()
    
    # 4. Save to disk
    for b in range(current_batch):
        # Calculate the actual sample index for saving
        sample_idx = z + b 
        
        os.makedirs(os.path.join(output_folder, str(sample_idx)), exist_ok=True)
        for t in range(num_lead):
            np.save(
                os.path.join(output_folder, str(sample_idx), f'{t}.npy'), 
                generated_samples_unnormlalized[b, t]
            )
            