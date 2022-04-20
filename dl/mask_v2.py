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

    frame_nr = 0
    fps = .0
    start = time.time()

    net = cv2.dnn.readNet('res10_300x300_ssd_iter_140000.caffemodel', "deploy.prototxt")
    webcam = cv2.VideoCapture(0)
    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1, 1)
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < .5:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (sx, sy, ex, ey) = box.astype("int")
            (sx, sy) = (max(0, sx), max(0, sy))
            (ex, ey) = (min(w - 1, ex), min(h - 1, ey))

            if sx == ex or sy == ey:
                continue

            face = frame[sy:ey, sx:ex]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = image.img_to_array(face)
            face = np.expand_dims(face, axis=0)
            face = preprocess_input(face)
            mask, no_mask = model.predict(face)[0]
            if mask > no_mask:
                color = (0, 255, 0)
                text = f'with mask {confidence:.2f}/{mask:.2f}'
            else:
                color = (0, 0, 255)
                text = f'without mask {confidence:.2f}/{no_mask:.2f}'
            cv2.rectangle(frame, (sx, sy), (ex, ey), color, 2)
            cv2.putText(frame, text, (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
