from cv2 import cv2
import numpy as np
import pathlib


def get_faces_data() -> tuple[np.ndarray, np.ndarray]:
    data_faces = []
    data_target = []
    data_folder = str(pathlib.Path(__file__).parent.resolve()) + "/faces/s"

    for i in range(1, 41):
        for j in range(1, 11):
            image = cv2.cvtColor(cv2.imread(data_folder + str(i) + "/" + str(j) + ".bmp"), cv2.COLOR_BGR2GRAY)
            data_faces.append(image / 255)
            data_target.append(i)

    return np.array(data_faces), np.array(data_target)


NUM_FACES_OF_PERSON_IN_DATASET = 10


def split_data(train_data: np.ndarray, target_data: np.ndarray, num_faces_for_train: int) -> tuple[
    list, list, list, list]:
    all_faces_train_chunks = np.array_split(train_data, len(train_data) / NUM_FACES_OF_PERSON_IN_DATASET)
    all_faces_target_chunks = np.array_split(target_data, len(train_data) / NUM_FACES_OF_PERSON_IN_DATASET)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for person_images, person_targets in zip(all_faces_train_chunks, all_faces_target_chunks):
        np.random.shuffle(person_images)

        x_train.extend(person_images[:num_faces_for_train])
        y_train.extend(person_targets[:num_faces_for_train])

        x_test.extend(person_images[num_faces_for_train:])
        y_test.extend(person_targets[num_faces_for_train:])

    return x_train, y_train, x_test, y_test

