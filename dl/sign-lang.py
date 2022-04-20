import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import os
from tensorflow.python.ops.confusion_matrix import confusion_matrix


MODEL_DIR = 'sign_lang_model'


def image_data_gen():
    return ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)


def image_data_batches(path, test=False):
    return image_data_gen().flow_from_directory(
        directory=path, target_size=(224, 224), batch_size=10, shuffle=not test)


def create_model():
    train_batches = image_data_batches('data/sign-lang/train')
    valid_batches = image_data_batches('data/sign-lang/valid')

    mobile = tf.keras.applications.mobilenet.MobileNet()
    x = mobile.layers[-6].output
    output = Dense(units=10, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=output)
    for layer in model.layers[:-23]:
        layer.trainable = False

    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=train_batches, validation_data=valid_batches, epochs=30, verbose=1)
    os.mkdir(MODEL_DIR)
    model.save(MODEL_DIR)
    return model


def main():
    if os.path.exists(os.path.join(MODEL_DIR, 'saved_model.pb')):
        model = tf.keras.models.load_model(MODEL_DIR)
    else:
        model = create_model()

    test_batches = image_data_batches('data/sign-lang/test', test=True)
    predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)
    cm = confusion_matrix(labels=test_batches.classes, predictions=np.argmax(predictions, axis=-1))
    print(cm)


if __name__ == '__main__':
    main()
