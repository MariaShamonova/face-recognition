import random

from cv2 import cv2
import numpy as np
import pathlib
import cvzone

def get_faces_data(path: str = '/faces', format: str = '.bmp', num_classes: int = 40, num_images: int = 10) -> tuple[np.ndarray, np.ndarray]:
    data_faces = []
    data_target = []
    data_folder = str(pathlib.Path(__file__).parent.resolve()) + path + "/s"

    for i in range(1, num_classes + 1):
        for j in range(1, num_images + 1):
            image = cv2.cvtColor(cv2.imread(data_folder + str(i) + "/" + str(j) + format), cv2.COLOR_BGR2GRAY)
            data_faces.append(image / 255)
            data_target.append(i)

    return np.array(data_faces), np.array(data_target)

def generate_mask_on_images(num_classes: int = 40, num_images: int = 40):
    front_img = cv2.imread("mask-icon.png", cv2.IMREAD_UNCHANGED)

    data_folder = str(pathlib.Path(__file__).parent.resolve())
    for i in range(1, num_classes + 1):
        for j in range(1, num_images + 1):

            back_img = cv2.imread(data_folder + f"/faces/s{i}/{j}.bmp")

            result_img = cvzone.overlayPNG(back_img, front_img, [10, 60])
            cv2.imwrite(data_folder + f'/masks/s{i}/{j}.png', result_img)



NUM_FACES_OF_PERSON_IN_DATASET = 10


def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

def split_data(train_data: np.ndarray,
               target_data: np.ndarray,
               data_deidentify: np.ndarray,
               data_mask: np.ndarray,
               num_faces_for_train: int,
               selected_faces_index: bool = 0) -> tuple[
    list, list, list, list]:

    data = create_data(train_data, target_data, data_deidentify, data_mask, selected_faces_index)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for elements in zip(*data):
        person_images = elements[0]
        person_targets = elements[1]
        if selected_faces_index == 1:
            deidentify_faces = elements[2]
        if selected_faces_index == 2:
            mask_faces = elements[2]

        if selected_faces_index == 1:
            person_images, data_deidentify = unison_shuffled_copies(person_images, data_deidentify)
        elif selected_faces_index == 2:
            person_images, data_mask = unison_shuffled_copies(person_images, data_mask)
        else:
            np.random.shuffle(person_images)

        x_train.extend(person_images[:num_faces_for_train])
        y_train.extend(person_targets[:num_faces_for_train])

        if selected_faces_index == 1:
            x_test.extend(deidentify_faces[num_faces_for_train:])
        elif selected_faces_index == 2:
            x_test.extend(mask_faces[num_faces_for_train:])
        else:
            x_test.extend(person_images[num_faces_for_train:])

        y_test.extend(person_targets[num_faces_for_train:])

    return x_train, y_train, x_test, y_test


def split_data_for_cross_validation(train_data: np.ndarray,
                                    target_data: np.ndarray,
                                    num_folds: int,
                                    data_deidentify: np.ndarray,
                                    data_mask: np.ndarray,
                                    selected_faces_index: bool = 0) -> tuple[
    list, list, list, list]:

    data = create_data(train_data, target_data, data_deidentify, data_mask, selected_faces_index)

    for chunk in data[0]:
        np.random.shuffle(chunk)

    faces_indexes = np.arange(NUM_FACES_OF_PERSON_IN_DATASET)

    np.random.shuffle(faces_indexes)

    split_faces_indexes = np.array_split(faces_indexes, num_folds)

    for test_indexes in split_faces_indexes:
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        train_indexes = set(faces_indexes) - set(test_indexes)

        for elements in zip(*data):
            person_images = elements[0]
            person_targets = elements[1]

            if selected_faces_index == 1:
                deidentify_faces = elements[2]
            if selected_faces_index == 2:
                mask_faces = elements[2]

            x_train.extend(person_images[index] for index in train_indexes)
            y_train.extend(person_targets[index] for index in train_indexes)

            if selected_faces_index == 1:
                x_test.extend(deidentify_faces[index] for index in test_indexes)
            elif selected_faces_index == 2:
                x_test.extend(mask_faces[index] for index in test_indexes)
            else:
                x_test.extend(person_images[index] for index in test_indexes)

            y_test.extend(person_targets[index] for index in test_indexes)

        yield x_train, y_train, x_test, y_test


def create_data(
                train_data: np.ndarray,
                target_data: np.ndarray,
                data_deidentify: np.ndarray,
                data_mask: np.ndarray,
                selected_faces_index: bool = 0 ):
    data = [np.array_split(train_data, len(train_data) / NUM_FACES_OF_PERSON_IN_DATASET),
            np.array_split(target_data, len(train_data) / NUM_FACES_OF_PERSON_IN_DATASET)]
    if selected_faces_index == 1:
        data.append(np.array_split(data_deidentify, len(data_deidentify) / NUM_FACES_OF_PERSON_IN_DATASET))
    elif selected_faces_index == 2:
        data.append(np.array_split(data_mask, len(data_mask) / NUM_FACES_OF_PERSON_IN_DATASET))

    return data
