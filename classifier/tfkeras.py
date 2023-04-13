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
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, required=True)
args = parser.parse_args()

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

with open("/content/drive/MyDrive/configs/polyp/configs.yml", 'r') as file:
    configs = yaml.safe_load(file)

configs = AttributeDict(configs)

batch_size = configs.BATCH_SIZE
IMG_SIZE = configs.IMAGE_SIZE
size = (IMG_SIZE, IMG_SIZE)

# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   configs.TRAIN_PATH,
#   image_size=size,
#   batch_size=batch_size)

# valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   configs.VALIDATION_PATH,
#   image_size=size,
#   batch_size=batch_size)

# class_names = list(train_ds.class_names)
# print(class_names)
# NUM_CLASSES = len(class_names)
# open("classes.txt", "w").write("\n".join(class_names))

# train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, depth=NUM_CLASSES)))
# valid_ds = valid_ds.map(lambda x, y: (x, tf.one_hot(y, depth=NUM_CLASSES)))

# class GCAdam(Adam):
#     def get_gradients(self, loss, params):
#         # We here just provide a modified get_gradients() function since we are
#         # trying to just compute the centralized gradients.

#         grads = []
#         gradients = super().get_gradients()
#         for grad in gradients:
#             grad_len = len(grad.shape)
#             if grad_len > 1:
#                 axis = list(range(grad_len - 1))
#                 grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
#             grads.append(grad)

#         return grads
    
def build_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    model = EfficientNetV2S(weights="/content/drive/MyDrive/kd_effi_v2s.h5", include_top=True, input_tensor=x, classes=2)
    outputs = model.output
    model = tf.keras.Model(inputs, outputs, name="kd_effi_v2s")
    
    # optimizer = GCAdam(learning_rate=1e-2)
    # model.compile(
    #     optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    # )
    return model

# callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model = build_model()

# epochs = 50
# hist = model.fit(train_ds, epochs=epochs, validation_data=valid_ds, callbacks=[callback], verbose=1)
# model.save("kd_effi_v2s.h5")

# true_categories = []
# predicted_categories = []
# false_prediction = dict()
# for x, y in valid_ds.unbatch():
#     true_categories.append(y.numpy().argmax())
#     predicted_categories.append(model.predict(tf.reshape(x, (1, IMG_SIZE, IMG_SIZE, 3))).argmax(axis=1)[0])
#     if true_categories[-1] != predicted_categories[-1]:
#         false_prediction[class_names[true_categories[-1]]] = [x.numpy(), class_names[predicted_categories[-1]]]

# confusion_matrix = confusion_matrix(true_categories, predicted_categories, labels=list(range(NUM_CLASSES)))

# # -*- coding: utf-8 -*-

# import seaborn as sn
# import pandas as pd
# import matplotlib.pyplot as plt
# from bidi import algorithm as bidialg

# labels = [bidialg.get_display(name) for name in class_names]

# df_cm = pd.DataFrame(confusion_matrix, index = labels, columns = labels)
# plt.figure(figsize = (3,3))
# sn.heatmap(df_cm, annot=True)
# plt.tight_layout()
# plt.savefig('confusion_matrix.png')


import innvestigate
import cv2, sys

tf.compat.v1.disable_eager_execution()
print(model.summary())
sys.exit()
model = innvestigate.model_wo_softmax(model)
analyzer = innvestigate.create_analyzer("deep_taylor", model)

img = cv2.imread(args.img)
inp = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)
rgb_tensor = tf.expand_dims(rgb_tensor , 0)
a = analyzer.analyze(rgb_tensor)
a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
a /= np.max(np.abs(a))

print("################################ yay #################################")
# Plot
plt.imshow(a[0], cmap="seismic", clim=(-1, 1))
plt.show()