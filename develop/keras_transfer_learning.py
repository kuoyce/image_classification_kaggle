# %%
import os

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras import layers, Sequential, Model, regularizers

import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# %%
folderpath = r'..\database\archive'

train_filepath = os.path.join(folderpath, 'seg_train', 'seg_train')
test_filepath = os.path.join(folderpath, 'seg_test', 'seg_test')

batch_size = 64 #32
img_height = 150
img_width = 150
seed = 42

# Load dataset using keras utils from path
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
  train_filepath,
  validation_split=0.2,
  subset="both",
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size,)

test_ds = tf.keras.utils.image_dataset_from_directory(
  test_filepath,
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  shuffle=False)

class_names = train_ds.class_names
print(class_names)

## preprocess data + prepare for faster loading
# create more variants in images to reduce overfitting and have a more robust model
data_augmentation = Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

# Apply `data_augmentation` to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)

# cache and prefetch the data to configure dataset for performance, more info can be found at https://www.tensorflow.org/guide/data_performance
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


# cache and prefetch the data to configure dataset for performance, more info can be found at https://www.tensorflow.org/guide/data_performance
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


# %%

## creating basic model
input_shape = (img_height, img_width, 3)
num_classes = len(class_names)
callbacks = [
    #tf.keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
]

# base_model = tf.keras.applications.VGG16( #6mins per epoch 88%acc]
# base_model = tf.keras.applications.InceptionV3( #2mins per epoch ~ 50%acc]
base_model = tf.keras.applications.ResNet50( #4mins per epoch ~ 90%acc]
    weights='imagenet', # Load weights pre-trained on IMageNet
    input_shape = input_shape,
    include_top=False # do no include ImageNet classifier on top
)

base_model.trainable = False

# base_outlayer = base_model.get_layer('block5_pool')
# base_outlayer = base_model.get_layer('conv5_block3_out')
# base_outlayer = base_model.get_layer('conv4_block6_out') 
base_outlayer = base_model.get_layer('conv3_block4_out')
base_output = base_outlayer.output

x = layers.Flatten()(base_output)
x = layers.Dense(64, activation='relu')(x)
x = layers.BatchNormalization()(x)
# x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu'
                 , kernel_regularizer=regularizers.l2(0.03)
                 , activity_regularizer=regularizers.l2(0.001)
                 )(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.7)(x)
x = layers.Dense(num_classes, activation='softmax', name="outputs")(x)

model = Model(base_model.input, x)

# model = Sequential([
#     #base_model,
#     base_model.get_layer('mixed9'),
#     layers.Flatten(),
#     layers.Dense(1024, activation='relu'),
#     layers.Dense(1024, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dropout(0.5),
#     layers.Dense(num_classes, activation='softmax', name="outputs")

# ])

optim = tf.keras.optimizers.Adam(learning_rate=0.0010) #default=0.001
#complie
model.compile(optimizer=optim,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


# %%
model.summary()

# %%
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

epochs_range = range(len(history.history['loss']))

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
# %%

#eval on test_ds
model.evaluate(test_ds)

#analyse model predictions on test dataset
predict_proba = model.predict(test_ds)
pred_labels = np.argmax(predict_proba, axis=1)

# test_labels = list(test_ds.unbatch().map(lambda x,y: y))
images, test_labels = tuple(zip(*list(test_ds.unbatch())))
cf_mtx = confusion_matrix(test_labels, pred_labels)

group_counts = ["{0:0.0f}".format(value) for value in cf_mtx.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_mtx.flatten()/np.sum(cf_mtx)]
box_labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
box_labels = np.asarray(box_labels).reshape(6, 6)

plt.figure(figsize = (12, 10))
sns.heatmap(cf_mtx, xticklabels=class_names, yticklabels=class_names,
           cmap="YlGnBu", fmt="", annot=box_labels)
plt.xlabel('Predicted Classes')
plt.ylabel('True Classes')
plt.show()

print(classification_report(test_labels, pred_labels, target_names=class_names))


##decide good number of epochs


# %%
def display_wrongly_predicted(real_labels,pred_labels,img_ds,num_samples=16):

    arr_wrong = (test_labels != pred_labels).nonzero()
    arr_wrong = [i for x in arr_wrong for i in x]
    rs = np.random.randint(len(arr_wrong), size=num_samples)
    selected_arr_wrong = [arr_wrong[x] for x in rs]

    fig, axes = plt.subplots(nrows=(int(np.ceil(num_samples/4))),ncols=4, figsize=(12,12))

    for c, ax in enumerate(axes.flatten()):
        if num_samples == c:
            break
        i = selected_arr_wrong[c]
        actual_label_name,pred_label_name = class_names[test_labels[i]], class_names[pred_labels[i]]
        arr = img_ds
        ax.imshow(arr[i]/255)
        ax.set_title("ID:"+str(i)+" Actual: "+ actual_label_name)
        ax.set_xlabel("Prediction: "+ pred_label_name)
    plt.tight_layout()
    plt.show()
    

display_wrongly_predicted(test_labels,pred_labels,images,num_samples=16)
# %%

def display_proba(real_label, proba, img, class_names=[]):
    assert tf.is_tensor(real_label)

    pred_label = np.argmax(proba, axis=0)
    if class_names ==[]:
        class_names = range(len(proba))
        actual_label_name,pred_label_name = real_label, pred_label
    else:
        actual_label_name,pred_label_name = class_names[real_label], class_names[pred_label]
    
    fig, axes = plt.subplots(2, 1, figsize=(5,4),gridspec_kw={'height_ratios': [3, 1]})
    axes[0].imshow(img/255)
    axes[1].bar(class_names,proba)
    axes[0].set_title('Actual: '+ actual_label_name)
    plt.xlabel('Pred: '+ pred_label_name)

    plt.tight_layout()
    plt.show()


rs = np.random.randint(len(test_labels), size=1)
rss = 2838 #rs[0]
real_label, proba, img = test_labels[rss], predict_proba[rss], images[rss]

display_proba(real_label, proba, img, class_names=class_names)
# %%
sns.histplot(predict_proba.flatten())
 # %%
