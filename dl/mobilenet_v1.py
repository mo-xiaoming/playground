import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
import os


mobile = tf.keras.applications.mobilenet.MobileNet()


def prepare_image(file):
    img_path = 'data/mobilenet-samples'
    img = image.load_img(os.path.join(img_path, file), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


def get_predict(path):
    preprocessed_image = prepare_image(path)
    predictions = mobile.predict(preprocessed_image)
    results = imagenet_utils.decode_predictions(predictions)
    print(path, results)


for i in ('1.jpg', '2.jpg', '3.jpg'):
    get_predict(i)