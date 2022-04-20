import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import os

from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.ops.confusion_matrix import confusion_matrix

MODEL_DIR = 'mask_v1_model'


def image_data_gen():
    return ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)


def image_data_batches(path, test=False):
    return image_data_gen().flow_from_directory(
        directory=path, target_size=(224, 224), batch_size=10, shuffle=not test)


def create_model():
    train_batches = image_data_batches('data/mask/data/train')
    valid_batches = image_data_batches('data/mask/data/valid')

    mobile = tf.keras.applications.mobilenet.MobileNet()
    x = mobile.layers[-6].output
    output = Dense(units=2, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=output)
    for layer in model.layers[:-23]:
        layer.trainable = False

    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=1)
    os.mkdir(MODEL_DIR)
    model.save(MODEL_DIR)
    return model


def main():
    if os.path.exists(os.path.join(MODEL_DIR, 'saved_model.pb')):
        model = tf.keras.models.load_model(MODEL_DIR)
    else:
        model = create_model()

    scale = 4

    frame_nr = 0
    fps = .0
    start = time.time()

    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    webcam = cv2.VideoCapture(0)
    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1, 1)
        mini = cv2.cvtColor(cv2.resize(frame, (frame.shape[1] // scale, frame.shape[0] // scale)), cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(mini)
        for face in faces:
            x, y, w, h = [v * scale for v in face]
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (224, 224))
            face = image.img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            mask, no_mask = model.predict(face)[0]
            color = (0, 255, 0) if mask > no_mask else (0, 0, 255)
            text = 'with mask' if mask > no_mask else 'without mask'
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        frame_nr += 1
        if frame_nr == 10:
            fps = 10 / (time.time() - start)
            frame_nr = 0
            start = time.time()
        cv2.putText(frame, f"{fps:.1f} fps", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1., (255, 0, 0), 2)

        cv2.imshow('LIVE', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()
    '''
    # to (224, 224)
    img = image.load_img('/Users/mx/Pictures/Photo on 12-25-20 at 19.33.jpg', target_size=(224, 224))
    # to (224, 224, 3), add channels
    img_array = image.img_to_array(img)
    # to (1, 224, 224, 3), add numbers of image
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    print(model.predict(img_array_expanded_dims))
    '''


if __name__ == '__main__':
    main()
