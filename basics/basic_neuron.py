# A neuron to formulate the y=2x-1
# The loss function is the MeanSquaredError & the optimizer function is STOCHASTIC GRADIENT DESCENT
import tensorflow as tf
import numpy as np
from tensorflow import keras

# neural network with one neuron,  1 layer, 1 input value
#dense indicates that it is a connected layer
model = tf.keras.Sequential(
    [keras.layers.Dense(units=1, input_shape=[1])]
)
print(f"Model shape is :: {model}")
model.compile(optimizer='sgd', loss='mean_squared_error')

x = tf.constant([-3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=tf.float32)
y = tf.constant([-7, -5, -3, -1, 1, 3, 5, 7, 9], dtype=tf.float32)

model.fit(x, y, epochs=1000)

for val in [20, 30,-12, -43, 0,500]:
    pred = model.predict([val])
    actual = (2*val) -1
    print(f"Prediction ---> : {pred} expected {actual}")