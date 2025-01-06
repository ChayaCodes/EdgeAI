import os
import datetime
import simple_nn_model as my_model

# Uncomment if you want CPU-only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from hf_dataset_adapter import get_train_test_dataset

train_dataset, test_dataset = get_train_test_dataset()

# 4. Build the model (ConvNet for MNIST)
model = my_model.get_model()

# 6. Train the model

# before training, let's log the progress:
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

model.fit(
    train_dataset,
    epochs=1,
    validation_data=test_dataset,
    callbacks=[tensorboard_callback]
)

# 7. Evaluate on test data
test_loss, test_acc = model.evaluate(test_dataset)
print(f"\nTest accuracy: {test_acc:.4f}")

# Save weights in TensorFlow's checkpoint format
model_dir = "my_kmodel"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model.save(f'model.keras')
SavedModel_dir = "my_SavedModel"
model.export(SavedModel_dir)
