import os, yaml
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
import tensorflow_datasets as tfds
from tqdm import tqdm
import numpy as np
import random
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

with open("cfg", 'r') as file:
    configs = yaml.safe_load(file)

configs = AttributeDict(configs)

batch_size = 32
IMG_SIZE = configs.IMAGE_SIZE
size = (IMG_SIZE, IMG_SIZE)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  configs.TRAIN_PATH,
  image_size=size,
  batch_size=batch_size)

valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
  configs.VALIDATION_PATH,
  image_size=size,
  batch_size=batch_size)

class_names = list(train_ds.class_names)
print(class_names)
NUM_CLASSES = len(class_names)
open("classes.txt", "w").write("\n".join(class_names))

train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, depth=NUM_CLASSES)))
valid_ds = valid_ds.map(lambda x, y: (x, tf.one_hot(y, depth=NUM_CLASSES)))

class GCAdam(Adam):
    def get_gradients(self, loss, params):
        # We here just provide a modified get_gradients() function since we are
        # trying to just compute the centralized gradients.

        grads = []
        gradients = super().get_gradients()
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads
    
def build_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    model = EfficientNetV2S(include_top=True, input_tensor=x)
    outputs = model.output

    model = tf.keras.Model(inputs, outputs, name="kd_effi_v2s")
    
    optimizer = GCAdam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

model = build_model()

epochs = 50
hist = model.fit(train_ds, epochs=epochs, validation_data=valid_ds, verbose=1)
model.save("kd_effi_v2s.h5")

true_categories = []
predicted_categories = []
false_prediction = dict()
for x, y in valid_ds.unbatch():
    true_categories.append(y.numpy().argmax())
    predicted_categories.append(model.predict(tf.reshape(x, (1, IMG_SIZE, IMG_SIZE, 3))).argmax(axis=1)[0])
    if true_categories[-1] != predicted_categories[-1]:
        false_prediction[class_names[true_categories[-1]]] = [x.numpy(), class_names[predicted_categories[-1]]]

confusion_matrix = confusion_matrix(true_categories, predicted_categories, labels=list(range(NUM_CLASSES)))

# -*- coding: utf-8 -*-

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from bidi import algorithm as bidialg

labels = [bidialg.get_display(name) for name in class_names]

df_cm = pd.DataFrame(confusion_matrix, index = labels, columns = labels)
plt.figure(figsize = (40,40))
sn.heatmap(df_cm, annot=True)
plt.tight_layout()
plt.savefig('confusion_matrix.png')