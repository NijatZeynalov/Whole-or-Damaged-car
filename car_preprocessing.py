#import modules
import os
import zipfile
import random
import tensorflow as tf
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd
import numpy as np

#building callbacks class
class myCallback(tf.keras.callbacks.Callback):
    # Your Code
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > 0.98):
            print("\nReached 98% accuracy so cancelling training!")
            self.model.stop_training = True
callbacks = myCallback()

#define categories
base_dir = 'cars'

train_dir = os.path.join(base_dir, 'training')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat/dog pictures
train_damage_dir = os.path.join(train_dir, '00-damage')
train_whole_dir = os.path.join(train_dir, '01-whole')

# Directory with our validation cat/dog pictures
validation_damage_dir = os.path.join(validation_dir, '00-damage')
validation_whole_dir = os.path.join(validation_dir, '01-whole')

train_damage_fnames = os.listdir( train_damage_dir )
train_whole_fnames = os.listdir( train_whole_dir )

print(train_damage_fnames[:10])
print(train_whole_fnames[:10])

#training and validating data generator
#train
TRAIN_DIR = 'cars\\training'
train_datagen = ImageDataGenerator(rescale=1.0/255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(TRAIN_DIR, batch_size= 120, class_mode='binary', target_size=(150, 150))

#validation
VALIDATION_DIR = 'cars\\validation'
validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, batch_size=120, class_mode='binary', target_size=(150, 150))
