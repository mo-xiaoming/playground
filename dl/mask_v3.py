import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

MODEL_DIR = 'model'
ROOT = './data/mask'
MODEL_DIR2 = 'model2'

BATCH_SIZE = 10
IMG_SIZE = (224, 224)

train_dataset = image_dataset_from_directory(
    os.path.join(ROOT, 'data/train'),
    label_mode='categorical',  # important, ignore this get binary_crossentropy
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE)
validation_dataset = image_dataset_from_directory(
    os.path.join(ROOT, 'data/valid'),
    label_mode='categorical',
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE)

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, classes=2)
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)

base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

'''
w = base_model.output
w = tf.keras.layers.GlobalAveragePooling2D()(w)
w = tf.keras.layers.Dense(128, activation="relu")(w)
output = tf.keras.layers.Dense(2, activation="softmax")(w)
model = tf.keras.Model(base_model.input, output)
'''
'''
x=base_model.output
x=tf.keras.layers.GlobalAveragePooling2D()(x)
x=tf.keras.layers.Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=tf.keras.layers.Dense(1024,activation='relu')(x) #dense layer 2
x=tf.keras.layers.Dense(512,activation='relu')(x) #dense layer 3
preds=tf.keras.layers.Dense(2,activation='softmax')(x) #final layer with softmax activation for N classes
model=tf.keras.Model(inputs=base_model.input,outputs=preds)
'''
'''
x = base_model.output
output = tf.keras.layers.Dense(units=2, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=output)
'''
# mobile.summary()
# x = mobile.layers[-1].output
# output = Dense(units=2, activation='softmax')(x)
# model = Model(inputs=mobile.input, outputs=output)
# for layer in model.layers[:-23]:
#    layer.trainable = False
model.summary()
initial_epochs = 10
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate), loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x=train_dataset, validation_data=validation_dataset, epochs=initial_epochs)

model_path = os.path.join(ROOT, MODEL_DIR)
os.mkdir(model_path)
model.save(model_path)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

base_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate / 10),
              metrics=['accuracy'])

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset)

model_path = os.path.join(ROOT, MODEL_DIR2)
os.mkdir(model_path)
model.save(model_path)

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs - 1, initial_epochs - 1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs - 1, initial_epochs - 1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)

'''
face_net = cv2.dnn_DetectionModel(weight_path, config_path)
face_net.SetInputSize(300, 300)
face_net.SetInputScale(1.0)
face_net.SetInputMean((104.0, 177.0, 123.0))
face_net.SetInputSwapRB(True)

class_ids, confidences, bbox = face_net.detect(frame, confThreshhold=.5)
for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), bbox):

'''
