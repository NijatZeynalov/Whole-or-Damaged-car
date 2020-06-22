#import modules
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, Conv2D, MaxPooling2D

#building model
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
model = tf.keras.models.Sequential(layers)

model.compile(loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
            metrics=['accuracy'])


#fitting model
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit_generator(train_generator, steps_per_epoch = 25, epochs = 25, validation_steps = 25, validation_data = validation_generator, callbacks = [early_stopping_cb])

model.save('car_model_final.h5')

model.save_weights('car_model_final.tf')

#plot model
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.show()
