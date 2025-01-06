# verify_model.py

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


def load_model(save_dir):
    model = TFAutoModelForSequenceClassification.from_pretrained(save_dir, from_pt=False)
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    return model, tokenizer


def test_inference(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="tf")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    print(f"Input text: {text}")
    print(f"Predicted class: {predicted_class}")


if __name__ == "__main__":
    save_dir = "my_tf_model"
    model, tokenizer = load_model(save_dir)
    test_text = "I love using TensorFlow Lite Micro on ESP32!"
    test_inference(model, tokenizer, test_text)
