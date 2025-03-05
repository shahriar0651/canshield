import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
import numpy as np


print("TensorFlow version:", tf.__version__)

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs Available: {len(gpus)}")
    for gpu in gpus:
        print(f"- {gpu}")
else:
    print("No GPUs found. TensorFlow will use the CPU.")



# Enable logging to see if operations are running on the GPU
tf.debugging.set_log_device_placement(True)

# Create a simple model
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(1000,)),
    keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Generate random data
X_train = np.random.randn(1000, 1000).astype(np.float32)
y_train = np.random.randint(0, 10, size=(1000,))

# Train the model
model.fit(X_train, y_train, epochs=1, batch_size=32)
