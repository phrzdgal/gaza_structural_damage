import os
import numpy as np
from sklearn.model_selection import train_test_split

# Creating a CNN - Buidling Blocks: Dense, Flatten, Conv2D and MaxPooling2D layers 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Likely to Switch to Custom/Personal Dataset in Future
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Building Model 
model = Sequential ([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)), 
    MaxPooling2D((2, 2)), 
    Conv2D(64, (3, 3), activation='relu'), 
    MaxPooling2D((2, 2)), 
    Flatten(), 
    Dense(64, activation='elu'), 
    Dense(10, activation='softmax')  
])

# Compiling Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', #categorical_crossentropy used for multi-class
    metrics=['accuracy']
)

# Training Model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
#Below used if we have custom datasets
#history = model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Evaluating Model Performance
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc}")

# Saving Model
model.save('image_classification_model.h5')

# Future testing with New Images
'''
loaded_model = load_model('image_classification_model.h5')

# Loading and Preprocessing Image
img = image.load_img('path/to/image.jpg', target_size=(32, 32))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediction
predictions = model.predict(img_array)
print(f"Predicted class: {np.argmax(predictions)}")
'''
