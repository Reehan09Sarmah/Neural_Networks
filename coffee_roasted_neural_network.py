import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam

# Important
# Choosing Correct Activation Function
# For Output Layer: depending on the type of output you are expecting from the neural network.
# - For Binary Classification Problem (yhat = 0 or 1) --> Always use Sigmoid Function
# - For Regression Problem (yhat can be -ve or +ve) --> Use Linear Activation Function
# - For Regression Problem (yhat can be non -ve only) --> Use ReLU
# For Hidden Layers
# - Mostly use ReLU (faster than Sigmoid)

# Design the model
model = Sequential([
    Dense(units=3, activation='relu'),
    Dense(units=1, activation='linear')
])

x = np.array([[200, 17],
              [120, 5],
              [425, 20],
              [212, 18]])
y = np.array([1, 0, 0, 1])

# compile the model with correct loss function
# use Adam Algorithm to handle learning rate automatically --> optimizes gradient descent
model.compile(optimizer=Adam(learning_rate=1e-3), loss=BinaryCrossentropy())
model.fit(x, y, epochs=100)  # Fit the dataset into the model

temperature = int(input('Enter temperature (in deg celsius)'))
duration = int(input('Enter duration (in mins)'))
percent = model.predict([[temperature, duration]])  # predict data

yhat = 0
ans = True

if percent >= 0.5:
    yhat = 1

if yhat == 0:
    ans = False

print(percent)
print(f'Coffee beans are well roasted? {ans}')