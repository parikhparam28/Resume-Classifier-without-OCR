#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# In[2]:


train_dir = Path("D:\\ResumeImages\\train")
test_dir = Path("D:\\ResumeImages\\test")


# In[3]:


train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 360,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.2,
    zoom_range = 0.1,
    fill_mode = 'constant',
)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[4]:


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode='binary',
    shuffle = True,
#     save_to_dir = train_dir,
#     save_prefix = 'augmented',
#     save_format = 'jpg'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=64,
    class_mode='binary',
    shuffle = True,
)


# In[5]:


model = Sequential ([
    Conv2D(32, (3,3), activation='relu',input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(256,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(512,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer = regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid'),
])


# In[6]:


model.compile(
    loss = BinaryCrossentropy(),
    optimizer = Adam(0.0002),
)

model.fit(
    train_generator,
    epochs = 25,
    validation_data = test_generator,
)


# In[7]:


evaluate = model.evaluate(test_generator)
print("Accuracy: ",evaluate)


# In[8]:


batch_X,Y_true = test_generator.next() 
Y_true = np.int_(Y_true)
predictions = model.predict(batch_X)
Y_pred = np.int_(predictions>=0.5)
Y_pred = np.squeeze(Y_pred)
print(Y_true)
print(Y_pred)


# In[9]:


conf_matrix = confusion_matrix(Y_true,Y_pred)

tn, fp, fn, tp = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

print("Confusion Matrix:", conf_matrix)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

