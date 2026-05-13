import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
    
import tensorflow as tf
from tensorflow import keras
from layers.denoiser import build_unet_model_c2 # The original 2D builder!
from layers.diffusion import GaussianDiffusion
from tensorflow.keras.models import load_model

pretrained_encoder_full = load_model('../saved_models/debug_encoder_cnn_patch_multivar.h5')
pretrained_encoder_full.summary()

# %%
bottleneck_layer = pretrained_encoder_full.get_layer('bottleneck').output
pretrained_encoder = tf.keras.Model(inputs=pretrained_encoder_full.input, 
                                    outputs=bottleneck_layer, 
                                    name='encoder')

for layer in pretrained_encoder.layers:
    layer.trainable = False

pretrained_encoder._name = 'encoder'


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

# 1. Build the exact 2D architecture
old_network = build_unet_model_c2(
    img_size_H=32, img_size_W=32, img_channels=5,
    widths=[64, 128, 256, 512], has_attention=[False, False, True, True],
    num_res_blocks=2, norm_groups=8, first_conv_channels=64,
    activation_fn=keras.activations.swish, encoder=pretrained_encoder
)
old_ema = build_unet_model_c2(
    img_size_H=32, img_size_W=32, img_channels=5,
    widths=[64, 128, 256, 512], has_attention=[False, False, True, True],
    num_res_blocks=2, norm_groups=8, first_conv_channels=64,
    activation_fn=keras.activations.swish, encoder=pretrained_encoder
)

# 2. Assemble the wrapper
old_model = DiffusionModel(network=old_network, ema_network=old_ema, gdf_util=GaussianDiffusion(1000), timesteps=1000)

# 3. Load the messy checkpoint
old_model.load_weights('../saved_models/codicast-patch/date/multivar/checkpoints/patch_2d_debug').expect_partial()

# 4. EXPORT THE GOLDEN WEIGHTS
old_model.network.save_weights('../saved_models/codicast-patch/date/multivar/checkpoints/patch_2d_debug_clean.h5')
print("Successfully extracted .h5 weights!")


