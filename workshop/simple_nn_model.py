import os

# Uncomment if you want CPU-only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow.keras import layers, models, callbacks

def get_model():
    model = models.Sequential([
        # First Conv2D
        layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1)),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Second Conv2D
        layers.Conv2D(filters=64, kernel_size=(3, 3)),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),

        # Dense layer
        layers.Dense(128),
        layers.Activation('relu'),

        # Dropout
        layers.Dropout(0.5),

        # Output layer for 10 classes
        layers.Dense(10),
        layers.Activation('softmax')
    ])

    # 5. Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
