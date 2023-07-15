import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # For splitting dataset
from sklearn.preprocessing import StandardScaler  # for Feature Scaling
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler  # for balancing out unequal data

df = pd.read_csv('./Datasets/diabetes.csv')

"""Plotting a Histogram for each column against their values separating people with and without diabetes"""

for i in range(len(df.columns[:-1])):
    label = df.columns[i]  # All the column names
    # add --> density=True, bins=15 to plot functions --> Normalize data.
    plt.hist(df[df['Outcome'] == 1][label], color='blue', label='Diabetes', alpha=0.7, density=True,
             bins=15)  # Diabetes positive patients
    plt.hist(df[df['Outcome'] == 0][label], color='red', label='No Diabetes', alpha=0.7, density=True,
             bins=15)  # Diabetes negative patients
    plt.title(label)
    plt.ylabel('Probability')
    plt.xlabel(label)
    plt.legend()
    plt.show()

# Divide the dataset into X and Y

x = df[df.columns[:-1]].values
y = df[df.columns[-1]].values

# Transform the data. Perform Feature Scaling.
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Just to create a dataframe and check the length of each type of data
data = np.hstack((x, np.reshape(y, (-1, 1))))
transformed_df = pd.DataFrame(data, columns=df.columns)
print(len(transformed_df[transformed_df['Outcome'] == 1]), len(transformed_df[transformed_df['Outcome'] == 0]))

# There's a huge difference between both types of data (with and without diabetes). So Balance them too.
over = RandomOverSampler()
x, y = over.fit_resample(x, y)

# Checking again
data = np.hstack((x, np.reshape(y, (-1, 1))))
transformed_df = pd.DataFrame(data, columns=df.columns)
print(len(transformed_df[transformed_df['Outcome'] == 1]), len(transformed_df[transformed_df['Outcome'] == 0]))

"""Split the dataset into training set and a temporary set.
Then split the temporary set into test set and a cross_validation set.


1.   test_set is required to test model's accuracy.
2.   cross_validation set is required to
choose the perfect model for the dataset checking the loss of each model. It tells us how well a model can handle new data.


"""

X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

# Create the model. Add layers and the number of neurons inside each. Also add the activation function for each neuron
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
]
)

# Compile the model. Use Optimizer algorithm, specify the loss function
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

"""Let's train it.
epochs = #iterations over the entire x and y data provided
validation_data:check the model performance after each epoch
"""

model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(X_valid, y_valid))

"""Lets Check the Loss and Accuracy."""

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {test_accuracy * 100.0}%")
