import os

import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf


model_dir = "my_model"
model = tf.saved_model.load(f'{model_dir}/my_model')

# Assume first layer is Conv2D
conv_layer = model.layers[0]
weights = conv_layer.get_weights()[0]  # 0 for the kernel, 1 for the bias
print("Kernel shape:", weights.shape)  # (3, 3, 1, 32) for example

num_filters = weights.shape[-1]
fig, axes = plt.subplots(4, 8, figsize=(10, 5))
axes = axes.flatten()

for i in range(num_filters):
    # Extract the filter
    f = weights[:, :, 0, i]  # if single channel
    # Normalize between 0 and 1 for display
    f_min, f_max = f.min(), f.max()
    if f_min < f_max:
        f = (f - f_min) / (f_max - f_min)
    axes[i].imshow(f, cmap='gray')
    axes[i].axis('off')
plt.show()
