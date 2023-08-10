import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.datasets import make_blobs

centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0, random_state=30)

print(X_train.shape, y_train.shape)

model = keras.Sequential([
    keras.layers.Dense(25, activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(4, activation='linear')
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(0.001)
)

model.fit(X_train, y_train, epochs=10)
pred = model.predict(X_train)

# Since last output layer was linear, it would give us z instead of softmax(z)
sm_pred = tf.nn.softmax(pred).numpy()
print(f"two example output vectors:\n {sm_pred[:2]}")
print("largest value", np.max(sm_pred), "smallest value", np.min(sm_pred))

# Check category prediction of first 5 examples trained on
# As output is the array of probabilities of each category, chose the category with the highest value.
for i in range(5):
    print(f"{pred[i]}, predicted category: {np.argmax(sm_pred[i])}, original category: {y_train[i]}")