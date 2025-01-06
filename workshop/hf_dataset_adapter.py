import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

from datasets import load_dataset
import numpy as np


def get_train_test_dataset(get_train=True, get_test=True, subset_train=None, subset_test=None):
    # We could download the dataset in the way TF is expecting by:
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # But this will miss the point of using HuggingFace... So we'll do it the longer way:

    # 1. Load the MNIST dataset from Hugging Face
    dataset = load_dataset("mnist")
    train_dataset, test_dataset = None, None

    # dataset is a dictionary-like object with train/test splits:
    # dataset["train"], dataset["test"]
    # 2. Preprocessing/Reshaping function
    #    MNIST images come as (28, 28) grayscale, but we want
    #    (28, 28, 1) and normalized [0,1] for a ConvNet.
    def preprocess(example):
        # Convert image from uint8 -> float32
        image = tf.cast(example["image"], tf.float32) / 255.0
        # Add channel dimension: (28, 28) -> (28, 28, 1)
        image = tf.expand_dims(image, axis=-1)
        example["image"] = image
        return example

    # 3. Convert to a tf.data.Dataset
    #    We'll do train/test splits separately for clarity.
    # Apply preprocessing and format to "tensorflow"
    if get_train:
        if subset_train is not None:
            dataset["train"] = dataset["train"].select(range(subset_train))

        dataset["train"] = dataset["train"].map(preprocess)

        train_dataset = dataset["train"].with_format("tensorflow").to_tf_dataset(
            columns=["image"],  # Features
            label_cols="label",  # Labels
            shuffle=True,
            batch_size=64
        )

    if get_test:
        if subset_test is not None:
            dataset["test"] = dataset["test"].select(range(subset_test))

        dataset["test"] = dataset["test"].map(preprocess)

        test_dataset = dataset["test"].with_format("tensorflow").to_tf_dataset(
            columns=["image"],
            label_cols="label",
            shuffle=False,
            batch_size=64
        )

    return train_dataset, test_dataset


def split_to_train_test_labels(data):
    test_x = []
    test_y = []

    # Iterate over the test_dataset and collect data
    for batch in data:
        x_batch, y_batch = batch
        test_x.append(x_batch.numpy())
        test_y.append(y_batch.numpy())

    # Concatenate all batches into single arrays
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)

    return test_x, test_y
