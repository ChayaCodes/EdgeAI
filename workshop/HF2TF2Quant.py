import tensorflow as tf
from transformers import TFAutoModelForImageClassification, AutoImageProcessor
import numpy as np
from PIL import Image
import requests
import io

##############################################################################
# 1. Download a ViT model from Hugging Face (TF-native)
##############################################################################
model_name = "google/vit-base-patch16-224"
print(f"Loading {model_name} in TensorFlow...")

# Use the image processor for ViT
image_processor = AutoImageProcessor.from_pretrained(model_name)

# This loads the TF version directly, no PyTorch needed
model = TFAutoModelForImageClassification.from_pretrained(model_name)

##############################################################################
# 2. Prepare an input image
##############################################################################
# For a real example, let's download a small test image from the internet
# (You can replace this with any local image file.)

url = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg"
)
response = requests.get(url)
image = Image.open(io.BytesIO(response.content)).convert("RGB")

# Convert to the correct size / format for ViT
inputs = image_processor(images=image, return_tensors="tf")

##############################################################################
# 3. Run inference in TF to confirm it works
##############################################################################
outputs = model(**inputs)
tf_logits = outputs.logits  # shape: (batch_size, num_classes)
pred_probs = tf.nn.softmax(tf_logits, axis=-1)[0].numpy()

# Just get top-5 for demonstration
top5_indices = pred_probs.argsort()[-5:][::-1]
print("===== TF Model Prediction (Top-5) =====")
for i in top5_indices:
    label_name = model.config.id2label[i]
    print(f"  Class {i} = {label_name} -> {pred_probs[i] * 100:.2f}%")


##############################################################################
# 4. Convert the model to a TensorFlow SavedModel
##############################################################################
saved_model_dir = "vit_saved_model"
model.export(saved_model_dir)

print(f"\nSavedModel created at: {saved_model_dir}")

##############################################################################
# 5. Convert to TFLite (Float32)
##############################################################################
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

tflite_model_path = "vit_float32.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Float32 TFLite model saved at: {tflite_model_path}")

##############################################################################
# 6. Apply Dynamic-Range Quantization
##############################################################################
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()

quantized_tflite_model_path = "vit_dynamic_quant.tflite"
with open(quantized_tflite_model_path, "wb") as f:
    f.write(quantized_tflite_model)

print(f"Dynamic-range quantized TFLite model saved at: {quantized_tflite_model_path}")


##############################################################################
# 7. Evaluate TFLite models vs. TF on the same image
##############################################################################
def run_tflite_inference(tflite_file, image_array):
    # image_array: shape (batch, 224, 224, 3)
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    # If needed, you can resize here for dynamic shape:
    #   interpreter.resize_tensor_input(input_details[0]["index"], image_array.shape)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]["index"], image_array)

    # Invoke
    interpreter.invoke()

    # Get output
    return interpreter.get_tensor(output_details[0]["index"])


# Convert the input to numpy
# The shape is (1, 224, 224, 3) in float32
image_input_np = inputs["pixel_values"].numpy()

# Run float32 TFLite inference
tflite_logits_float = run_tflite_inference(tflite_model_path, image_input_np)
pred_probs_float = tf.nn.softmax(tflite_logits_float, axis=-1)[0].numpy()

# Run quantized TFLite inference
tflite_logits_quant = run_tflite_inference(quantized_tflite_model_path, image_input_np)
pred_probs_quant = tf.nn.softmax(tflite_logits_quant, axis=-1)[0].numpy()

print("\n===== Comparison =====")
print("TF logits (first 5):", tf_logits[0, :5].numpy())
print("TFLite float logits (first 5):", tflite_logits_float[0, :5])
print("TFLite quant logits (first 5):", tflite_logits_quant[0, :5])

# Just for a final top-5 check
top5_tf = pred_probs.argsort()[-5:][::-1]
top5_tflite_float = pred_probs_float.argsort()[-5:][::-1]
top5_tflite_quant = pred_probs_quant.argsort()[-5:][::-1]

print("\nTop-5 classes by each model:")
print("TF Model:           ", top5_tf)
print("TFLite Float Model: ", top5_tflite_float)
print("TFLite Quant Model: ", top5_tflite_quant)

print("\nScript completed successfully!")
