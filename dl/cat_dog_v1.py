import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.python.ops.confusion_matrix import confusion_matrix


def image_data_gen():
    return ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input)


def image_data_batches(path, test=False):
    return image_data_gen().flow_from_directory(
        directory=path,
        target_size=(224, 224),
        classes=['cat', 'dog'],
        batch_size=10)


def train_model():
    train_batches = image_data_batches('data/dogs-vs-cats/train/train')
    valid_batches = image_data_batches('data/dogs-vs-cats/train/valid')

    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(units=2, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        x=train_batches,
        steps_per_epoch=len(train_batches),
        validation_data=valid_batches,
        validation_steps=len(valid_batches),
        epochs=10,
        verbose=1
    )
    model.save('cat_dog_v1_model')
    return model


def main():
    if os.path.exists('cat_dog_v1_model/saved_model.pb'):
        model = keras.models.load_model('cat_dog_v1_model')
    else:
        model = train_model()

    test_batches = image_data_batches('data/dogs-vs-cats/train/test', test=True)
    predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)
    cm = confusion_matrix(labels=test_batches.classes, predictions=np.argmax(predictions, axis=-1))
    print(cm)


if __name__ == '__main__':
    main()