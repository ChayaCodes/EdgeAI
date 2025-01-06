import os
import numpy as np
from hf_dataset_adapter import get_train_test_dataset, split_to_train_test_labels
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

# Load the TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path="model_quant_default.tflite")
interpreter = tf.lite.Interpreter(model_path="model_quant_full_int_with_io.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Evaluate on your test dataset (pseudo-example)
_ , test_dataset = get_train_test_dataset(get_train=False, get_test=True, subset_train=None, subset_test=1000)
test_data, test_labels = split_to_train_test_labels(test_dataset)

correct = 0
total = len(test_data)

for i in range(total):
    # Preprocess test_data[i] if needed:
    #   - Reshape to match input_details[0]['shape']
    #   - Scale or cast to int8 if your model expects int8 input

    input_data = test_data[i].reshape(input_details[0]['shape'])

    # If model input is int8:
    input_data = (input_data * 127).astype(np.int8)  # example scaling to [-128, 127]

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Post-process output to get predicted label
    predicted_label = np.argmax(output_data)
    if predicted_label == test_labels[i]:
        correct += 1

accuracy = correct / total
print("Quantized model accuracy:", accuracy)
