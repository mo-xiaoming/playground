import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import os
from tensorflow import keras
from tensorflow.keras.optimizers import Adam


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

MODEL_DIR = 'side_effects_model'


def create_data(out_liner, normal):
    labels = []
    samples = []

    for i in range(out_liner):
        # Younger individuals who did experience side effects
        random_younger = randint(13,64)
        samples.append(random_younger)
        labels.append(1)

        # Older individuals who did not experience side effects
        random_older = randint(65,100)
        samples.append(random_older)
        labels.append(0)

    for i in range(normal):
        # Younger individuals who did not experience side effects
        random_younger = randint(13,64)
        samples.append(random_younger)
        labels.append(0)

        # Older individuals who did experience side effects
        random_older = randint(65,100)
        samples.append(random_older)
        labels.append(1)

    return shuffle(np.array(samples), np.array(labels))


def scale_data(samples):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(samples.reshape(-1, 1))


def train_model():
    train_samples, train_labels = create_data(50, 1000)
    scaled_train_samples = scale_data(train_samples)
    model = keras.models.Sequential([
        keras.layers.Dense(units=16, input_shape=(1,), activation='relu'),
        keras.layers.Dense(units=32, activation='relu'),
        keras.layers.Dense(units=2, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    model.fit(
        x=scaled_train_samples,
        y=train_labels,
        validation_split=.2,
        batch_size=10,
        epochs=30,
        verbose=2)
    model.save(MODEL_DIR)
    return model


def main():
    if not os.path.exists(os.path.join(MODEL_DIR, 'saved_model.pb')):
        model = train_model()
    else:
        model = keras.models.load_model(MODEL_DIR)
    model.summary()
    test_samples, test_labels = create_data(10, 200)
    predictions = model.predict(x=scale_data(test_samples), batch_size=10, verbose=0)
    rounded_predictions = np.argmax(predictions, axis=1)
    misses = 0
    for i in zip(test_samples, test_labels, rounded_predictions, predictions):
        if i[1] != i[2]:
            print(i)
            misses += 1
    print(f"{misses} misses, {1 - misses / 210:.2f} accuracy")


if __name__ == '__main__':
    main()
