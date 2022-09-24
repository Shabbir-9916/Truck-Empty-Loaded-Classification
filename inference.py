import os
import random
import shutil
import matplotlib.pyplot as plt
from collections import Counter
from datetime import date
import numpy as np
from PIL import ImageFile
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import (ResNet50, preprocess_input)
from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img


base_model = ResNet50(include_top=False, weights='imagenet', input_shape = (224,224,3))

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)       #Additional
x = layers.Dense(512, activation='relu')(x)       #Additional
x = layers.Dense(256, activation='relu')(x)       #Additional
x = layers.Dense(256, activation='relu')(x)       #Additional

out = layers.Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=out)

for layer in base_model.layers:
  layer.trainable = False

model = Model(inputs=base_model.input, outputs=out)
model.load_weights('Sep-23-2022/')


model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=[
              keras.metrics.Precision(name='precision'),
              keras.metrics.Recall(name='recall'),
              keras.metrics.AUC(name='auc'),
              tfa.metrics.F1Score(name='f1_score', num_classes = 2)
              ]) 

image_path = 'truckEmptyLoaded12_77.jpg'


img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_preprocessed = preprocess_input(img_batch)
prediction = model.predict(img_preprocessed)[0]
id = np.argmax(prediction)
score = prediction[id]

classes =  {'Empty': 0, 'Loaded': 1}

val_list, key_list = list(classes.values()), list(classes.keys())
position = val_list.index(id)
pred = key_list[position]

print('Prediction: ', key_list[position], '; Confidence: ', score)


# In[ ]:




