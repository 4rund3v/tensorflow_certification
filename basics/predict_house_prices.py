from tensorflow import keras
import tensorflow as tf
'''
housing prices are priced at 50K (base) + 50k per bedroom
i.e 1bedroom house = 100k
    2bedroom house = 150k 
    Normalize the data to small values ( from 100k to 1,2)
    
'''


def house_model(y_new):
    xs = tf.constant([0,   1,  2,  3,  4, 5,   6,  7,  8,  9, 10], dtype=tf.float32)
    ys = tf.constant([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5], dtype=tf.float32)
    model = tf.keras.Sequential([
        keras.layers.Dense(units=1, input_shape=[1])
    ])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=500)
    return model.predict(y_new)[0]

prediction = house_model([12.0])
print(prediction)