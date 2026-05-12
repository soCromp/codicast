import numpy as np
import glob
import os

class KerasHybridNormalizer:
    def __init__(self, channel_names):
        """
        channel_names: list of strings, e.g., ['z500', 't850', 't2m', 'u10', 'v10']
        """
        self.channels = channel_names
        self.stats = {} 

    def fit(self, data, clamp=True, all_linear=True):
        """
        Expects data in (N, H, W, C) format.
        Computes per-channel 1st and 99th percentiles.
        """
        print("Computing channel statistics...")
        for i, char in enumerate(self.channels):
            # Extract single channel and flatten
            data_c = data[..., i].flatten()
            valid_mask = np.isfinite(data_c)
            arr = data_c[valid_mask]

            # --- LOGIC SWITCH ---
            if not all_linear and char.lower() in ['q', 'humidity', 'precip', 'windmag', 'wind-mag']:
                # Force strictly non-negative before log transform
                arr = np.log1p(np.maximum(arr, 0.0))
                method = 'log_robust'
            else: 
                method = 'linear_robust'
                
            if clamp:
                min_char = np.percentile(arr, 1)
                max_char = np.percentile(arr, 99)
            else:
                min_char = np.min(arr)
                max_char = np.max(arr)
            
            self.stats[char] = {
                'min': min_char,
                'max': max_char,
                'median': np.median(arr),
                'method': method
            }
            
            print(f"Var {char} ({method}): Min={min_char:.3f}, Max={max_char:.3f}, #Nans={np.sum(~np.isfinite(data_c))}")

    def normalize(self, x):
        """
        Expects x in (N, H, W, C) format. Returns normalized x in [-1, 1].
        """
        x_out = x.copy().astype(np.float32)
            
        for i, char in enumerate(self.channels):
            stat = self.stats[char]
            val = x_out[..., i]
            
            # Fill nans with median value
            val = np.nan_to_num(val, nan=float(stat['median']))
            
            # Apply Log if needed
            if stat['method'] == 'log_robust':
                val = np.maximum(val, 0.0) # Correct negative rounding errors
                val = np.log1p(val) 
            
            # Robust Scale to [-1, 1]
            scale = stat['max'] - stat['min']
            val = 2.0 * (val - stat['min']) / (scale + 1e-6) - 1.0
            
            # Clamp (Safety)
            val = np.clip(val, -1.0, 1.0)
                
            # Assign back
            x_out[..., i] = val
                
        return x_out

    def denormalize(self, x):
        """ Reverses operations. Expects x in (N, H, W, C) format. """
        x_out = x.copy().astype(np.float32)
            
        for i, char in enumerate(self.channels):
            stat = self.stats[char]
            val = x_out[..., i]
            
            # Inverse Scale
            scale = stat['max'] - stat['min']
            val = (val + 1.0) / 2.0 * scale + stat['min']
            
            # Inverse Log if needed
            if stat['method'] == 'log_robust':
                val = np.expm1(val)
                val = np.maximum(val, 0.0) # Physical constraint
            
            x_out[..., i] = val
                
        return x_out
    
    