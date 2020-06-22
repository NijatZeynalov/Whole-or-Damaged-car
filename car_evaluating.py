#import modules

import cv2
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, Conv2D, MaxPooling2D

#evaluate model
layers = [
    tf.keras.layers.Conv2D(32, 3, 3, input_shape=(150, 150, 3), activation='relu'),
    tf.keras.layers.Conv2D(32, 3, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(1, activation='sigmoid')
]
eval_model = tf.keras.models.Sequential(layers)


eval_model.load_weights('car_model_final.tf')
path = 'sell.jpg'
img = image.load_img(path, target_size=(150, 150))
img2 = image.load_img(path)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = eval_model.predict(images, batch_size=10)
print(classes[0])
prediction = ''
if classes[0]>0.5:
    prediction ='Whole'
else:
    prediction = 'Damaged'

#plot the test results

plt.imshow(img)
plt.title(prediction)
plt.show()