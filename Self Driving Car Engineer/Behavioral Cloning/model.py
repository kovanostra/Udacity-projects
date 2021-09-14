import os
import csv
import cv2
import argparse

import numpy as np

from typing import Tuple, List, Iterable

from matplotlib import pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from keras import layers, models, Model
from keras.callbacks import ModelCheckpoint


class CloningModel:
    def __init__(self, data_path: str, validation_size: float, test_size: float, batch_size: float,
                 epochs: int) -> None:
        self.data_path = data_path
        self.validation_size = validation_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.training_set = None
        self.validation_set = None
        self.image_shape = None
        self.training_generator = None
        self.validation_generator = None
        self.history = None
        self.test_loss = None

    def prepare_dataset(self) -> None:
        self._split_train_val_and_test_sets()
        self._get_image_dimensions(sample_data=self.training_set[0])
        self.training_generator = self._map_dataset_to_generator(dataset=self.training_set)
        self.validation_generator = self._map_dataset_to_generator(dataset=self.validation_set)
        self.test_generator = self._map_dataset_to_generator(dataset=self.test_set)

    def train(self) -> None:
        row, col, ch = self.image_shape
        model = models.Sequential()
        model.add(layers.Lambda(lambda x: x / 255 - 0.5, input_shape=(row, col, ch)))
        model.add(layers.Cropping2D(cropping=((50, 20), (0, 0))))
        model.add(layers.Conv2D(16, (5, 5), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(1))
        model.compile(loss='mse', optimizer='adam')
        best_model_save = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss')
        self.history = model.fit_generator(self.training_generator,
                                           steps_per_epoch=np.ceil(len(self.training_set) / self.batch_size),
                                           validation_data=self.validation_generator,
                                           validation_steps=np.ceil(len(self.validation_set) / self.batch_size),
                                           epochs=self.epochs,
                                           verbose=1,
                                           callbacks=[best_model_save])
        self.test_loss = model.evaluate_generator(self.test_generator,
                                                  steps=np.ceil(len(self.test_set) / self.batch_size))
        print("test_loss: {:.4f}".format(self.test_loss))

    def save_history(self) -> None:
        try:
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.plot(len(self.history.history['val_loss']) - 1, self.test_loss,
                     'o')  # plot x and y using blue circle markers
            plt.title('Loss per epoch')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['train_loss', 'val_loss', 'test_loss'], loc='upper right')
            plt.savefig("./training_history.png")
        except:
            print("Failed to save history! Are you connected to a GPU?")

    def _split_train_val_and_test_sets(self) -> None:
        all_driving_log_rows = []
        with open(os.path.join(self.data_path, 'driving_log.csv'), 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            next(csv_reader, None)
            for row in csv_reader:
                all_driving_log_rows.append(row)
        self.training_set, validation_and_test_set = train_test_split(all_driving_log_rows,
                                                                      test_size=self.validation_size + self.test_size)
        self.validation_set, self.test_set = train_test_split(validation_and_test_set, test_size=self.test_size)

    def _get_image_dimensions(self, sample_data: List[str]) -> None:
        sample_image = cv2.imread(os.path.join(self.data_path, sample_data[0]))
        self.image_shape = sample_image.shape

    def _map_dataset_to_generator(self, dataset: list) -> Iterable[np.ndarray]:
        while True:
            for dataset_index in range(0, len(dataset), self.batch_size):
                batch = dataset[dataset_index: dataset_index + self.batch_size]
                images = []
                measurements = []
                for sample in batch:
                    center_image = cv2.imread(os.path.join(self.data_path, sample[0].strip()))
                    left_image = cv2.imread(os.path.join(self.data_path, sample[1].strip()))
                    right_image = cv2.imread(os.path.join(self.data_path, sample[2].strip()))
                    images.extend([center_image, left_image, right_image])
                    measurement = float(sample[3].strip())
                    correction = 0.12
                    measurements.extend([measurement, measurement + correction, measurement - correction])
                X_train = np.stack(images, axis=0)
                y_train = np.array(measurements)
                yield sklearn.utils.shuffle(X_train, y_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('--data_path', type=str, default="./CarND-Behavioral-Cloning-P3/data/stage_1",
                        help='Path to the data directory.')
    parser.add_argument('--validation_size', type=float, default=0.2, help='Size of the validation set.')
    parser.add_argument('--test_size', type=float, default=0.1, help='Size of the test set.')
    parser.add_argument('--batch_size', type=float, default=8, help='Size of the batch.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs.')
    args = parser.parse_args()

    cloning_model = CloningModel(args.data_path, args.validation_size, args.test_size, args.batch_size, args.epochs)
    cloning_model.prepare_dataset()
    cloning_model.train()
    cloning_model.save_history()
