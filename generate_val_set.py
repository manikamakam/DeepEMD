import matplotlib.pyplot as plt
import numpy as np
import os
from keras import layers
import keras
from PIL import Image
import cv2 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 500

IMAGE_PATH = "/home/sri.makam@faradayfuture.com/Desktop/Gordian-TakeHome/data/images"
list = os.listdir(IMAGE_PATH)

for dat in list:
    path = os.path.join(IMAGE_PATH, dat)

    # Maximum Rotation
    max_rot = 80

    # ImageDataGenerator
    datagen_rot = ImageDataGenerator(rotation_range=max_rot, fill_mode='nearest')
    datagen_brightness = ImageDataGenerator(brightness_range=[0.8,1.2], fill_mode='nearest')
    datagen_zoom = ImageDataGenerator(zoom_range=[0.3,1.2])
    
    # Load Image
    img = Image.open(path)
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)

    # Rotation Iterator
    aug_rot = datagen_rot.flow(img, batch_size=1)
    result = next(aug_rot)[0].astype('uint8')
    result = np.expand_dims(result, axis=0)

    # Brightness iterator
    aug_bright = datagen_brightness.flow(result, batch_size=1)
    result = next(aug_bright)[0].astype('uint8')
    result = np.expand_dims(result, axis=0)

    # Zoom iterator
    aug_zoom = datagen_brightness.flow(result, batch_size=1)
    result = next(aug_zoom)[0].astype('uint8')
  
    print("../data/val/" + dat)
    plt.imsave("../data/val/" + dat, result)