import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

from hf_dataset_adapter import get_train_test_dataset, split_to_train_test_labels


def loop_based_inference(model, x_test, y_test):
    correct_predictions = 0
    total_samples = x_test.shape[0]

    for i in range(total_samples):
        # Expand dims from (28, 28, 1) to (1, 28, 28, 1) to make it a "batch"
        x_input = np.expand_dims(x_test[i], axis=0)

        # Get model prediction
        logits = model(x_input, training=False)  # shape (1, 10)

        # Find predicted class
        predicted_class = np.argmax(logits[0])

        # Compare with ground truth
        if predicted_class == y_test[i]:
            correct_predictions += 1

    return correct_predictions / total_samples


if __name__ == '__main__':
    model = tf.keras.models.load_model('model.keras')

    _, test_dataset = get_train_test_dataset(get_train=False, get_test=True,
                                             subset_train=None, subset_test=10)

    test_x, test_y = split_to_train_test_labels(test_dataset)

    accuracy = loop_based_inference(model, test_x, test_y)
    print(f"Manual Loop Test Accuracy: {accuracy:.4f}")

    # Now auto check teh same thing:
    loss, accuracy = model.evaluate(test_x, test_y, verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
