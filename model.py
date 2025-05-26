import tensorflow as tf
import keras 
import numpy as np
import matplotlib.pyplot as plt


# Data Loading

(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()


# Model definition

model = keras.Sequential([
        keras.layers.Flatten(28,28),
        keras.layers.Dense(128,activation='relu'),
        keras.layers.Dense(128,activation='relu'),
        keras.layers.Dense(10,activation='softmax')
])

# model compilation

model.compile(optimizer='adam',metrics=['accuracy'])

# model training

History = model.fit(x_train,y_train,batch_size=32,epochs=3)

# model evaluation

plt.plot(History.history['loss'], lable="Training loss")
plt.ylabel("Loss")
plt.xlabel("epochs")
plt.show()
