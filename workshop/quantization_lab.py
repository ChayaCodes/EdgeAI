import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import numpy as np
from hf_dataset_adapter import get_train_test_dataset

SavedModel_dir = "my_SavedModel"

# Initialize the converter
converter = tf.lite.TFLiteConverter.from_saved_model(SavedModel_dir)

# Post-Training Quantization:
# Must have line for any quantization:
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 1. Default: Dynamic Range Quantization
#       weights -> int8
#       activations remain float

# 2. For float16 quantization:
#       weights to float16 (more accurate. less compression. Good for GPU / ARM half-float acceleration etc.)
#       activations remain float
# converter.target_spec.supported_types = [tf.float16]

# 3. Full Integer Quantization
#   both weights and activation funcs
#   best option for microcontrollers
#   requires representative dataset
#
train_dataset, _ = get_train_test_dataset(get_train=True, get_test=False, subset_train=1000, subset_test=None)


def representative_data_gen():
    for x_batch, _ in train_dataset.take(100):
        x_batch = x_batch.numpy()  # shape (batch_size, 28, 28, 1)
        for i in range(x_batch.shape[0]):
            single_image = x_batch[i: i + 1]  # shape (1, 28, 28, 1)
            yield [single_image]


# Set the representative dataset
converter.representative_dataset = representative_data_gen

# Force integer quantization for all operators
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# 
# # Optional: Set i/o type to avoid floats at all:
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8

# Convert
tflite_model = converter.convert()

# Save the converted model to disk
with open("model_quant_full_int_with_io.tflite", "wb") as f:
    f.write(tflite_model)
