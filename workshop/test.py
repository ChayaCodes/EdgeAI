# quantize_model.py

import tensorflow as tf
from transformers import AutoTokenizer
from datasets import load_dataset


def representative_data_gen():
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    # Load a small subset of a dataset for calibration
    dataset = load_dataset("glue", "sst2", split="validation[:100]")  # Using first 100 samples
    for sample in dataset:
        text = sample['sentence']
        inputs = tokenizer(text, return_tensors="tf", padding='max_length', truncation=True, max_length=128)
        # Yield a dictionary of input tensors
        yield {key: tf.constant(value.numpy()) for key, value in inputs.items()}


def convert_and_quantize(saved_model_dir, output_tflite_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Set the representative dataset for full integer quantization
    converter.representative_dataset = representative_data_gen

    # Ensure that all activations are quantized to int8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Convert the model
    tflite_quant_model = converter.convert()

    # Save the quantized model
    with open(output_tflite_path, "wb") as f:
        f.write(tflite_quant_model)

    print(f"Quantized TFLite model saved to {output_tflite_path}")


if __name__ == "__main__":
    saved_model_dir = "my_tf_model"
    output_tflite_path = "model_quant.tflite"
    convert_and_quantize(saved_model_dir, output_tflite_path)
