import numpy as np
import os
import tifffile as tiff



AC = '/mnt/images/processed/2024_11_04-CTRL/AC.tif'
try:
    ac_image = tiff.imread(AC)
    print(f"AC image shape: {ac_image.shape}")
    print(f"AC image dtype: {ac_image.dtype}")
except Exception as e:
    print(f"Error reading AC image: {e}")