from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
# instead you can use --> import tensorflow as tf, then use tf.keras.

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# MNIST contains a collection of 70,000, 28 x 28 images of handwritten digits from 0 to 9
# We need 0 and 1 only. So lets filter it
x_train_filtered = x_train[(y_train == 0) | (y_train == 1)]
y_train_filtered = y_train[(y_train == 0) | (y_train == 1)]
x_test_filtered = x_test[(y_test == 0) | (y_test == 1)]
y_test_filtered = y_test[(y_test == 0) | (y_test == 1)]

# Build model
model = Sequential([
    # Flatten --> Convert multi-D array into 1-D array
    Flatten(input_shape=(28, 28)),  # images in mnist are 28 X 28
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

# Compile model --> use adam algo for optimised learning rate
model.compile(optimizer=Adam(learning_rate=1e-3), loss=BinaryCrossentropy(), metrics=['accuracy'])

# Train model
model.fit(x_train_filtered, y_train_filtered, epochs=5)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test_filtered, y_test_filtered)
print(f"Test Accuracy: {round(test_acc * 100.0, 3)}%")
print(f"Loss: {test_loss * 100.0}%")