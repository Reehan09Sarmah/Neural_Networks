"""
We are going to train a neural network dataset containing 28X28 sized
matrices that represent an image. The values inside the matrix are color pixel
values (0-255). They have outputs ranging from 0-9. Each number is a label
representing what the image is of. for eg; label 0 --> Ankle boot, label 8
--> Shirt, etc. So we'll train the model to recognize such images.

Sol: For the layers, input layer = a 1D array (convert the 28x28 matrix)
hidden layer: 128 neurons with ReLU activation.
Output layer: 10 neurons with softmax activation.
This is a classification problem but number of categories is 10. Identify
the labels an image belongs to.

"""

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

# labels --> Each image is mapped to a single label. 10 possibilities
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Every image -> 28 x 28 matrix, each value = pixel value (0-255)
# So let's make it small, within a certain range
train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # to convert matrix into a 1D array
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)

prediction = model.predict(test_images)

# prediction for 1 image = array of 10 values.
# Each value is a probability of what the model thinks the input is
# So we are going to take the maximum value of that array and associate its
# index with the class names.

# let's check the prediction for 1st 5 images of test set
for i in range(1, 3):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(f"Actual: index = {test_labels[i]} -> {class_names[test_labels[i]]}")
    plt.title(f"Prediction: {class_names[np.argmax(prediction[i])]}")  # argmax returns index of the max value of the array
    plt.show()