import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

print('keras is in!')

AUTOTUNE = tf.data.AUTOTUNE

folderpath = r'.\database\archive'

train_filepath = os.path.join(folderpath, 'seg_train', 'seg_train')
test_filepath = os.path.join(folderpath, 'seg_test', 'seg_test')

batch_size = 32
img_height = 150
img_width = 150
seed = 42

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
  train_filepath,
  validation_split=0.2,
  subset="both",
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

# create more variants in images to reduce overfitting and have a more robust model
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    #layers.RandomZoom(0.1),
  ]
)

# Apply `data_augmentation` to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=AUTOTUNE,
)

# cache and prefetch the data to configure dataset for performance, more info can be found at https://www.tensorflow.org/guide/data_performance
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# creating basic model
input_shape = (img_height, img_width, 3)
num_classes = len(class_names)
callbacks = [
    #keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    #tf.keras.callbacks.EarlyStopping(patience=2)
]

#build model
model = Sequential([
  layers.Rescaling(1./255, input_shape=input_shape),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.1),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.1),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.1),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, name="outputs")
])

#complie
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#view model summary
model.summary()

#train model
epochs=20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=callbacks
)

#display training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 9))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()