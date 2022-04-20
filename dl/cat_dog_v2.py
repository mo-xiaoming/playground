import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.python.ops.confusion_matrix import confusion_matrix

MODEL_DIR = 'cat_dog_v2_model'


def image_data_gen():
    return ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input)


def image_data_batches(path, test=False):
    return image_data_gen().flow_from_directory(
        directory=path,
        target_size=(224, 224),
        classes=['cat', 'dog'],
        batch_size=10, shuffle=not test)


def create_model():
    vgg16_model = tf.keras.applications.vgg16.VGG16()
    model = Sequential()
    for layer in vgg16_model.layers[:-1]:
        model.add(layer)
    for layer in model.layers:
        layer.trainable = False
    model.add(Dense(units=2, activation='softmax'))
    train_batches = image_data_batches('data/dogs-vs-cats/train/train')
    valid_batches = image_data_batches('data/dogs-vs-cats/train/valid')
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        x=train_batches,
        steps_per_epoch=len(train_batches),
        validation_data=valid_batches,
        validation_steps=len(valid_batches),
        epochs=5,
        verbose=1
    )
    model.save(MODEL_DIR)
    return model


def main():
    if os.path.exists(os.path.join(MODEL_DIR, 'saved_model.pb')):
        model = keras.models.load_model(MODEL_DIR)
    else:
        model = create_model()

    test_batches = image_data_batches('data/dogs-vs-cats/train/test', test=True)
    predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)
    cm = confusion_matrix(labels=test_batches.classes, predictions=np.argmax(predictions, axis=-1))
    print(cm)


if __name__ == '__main__':
    main()
