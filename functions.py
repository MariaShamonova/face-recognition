import cv2 as cv
import os
from pathlib import Path
import random as rnd
import numpy as np
PATH = Path(__file__).parent.absolute()


def mesh_data(data):
    indexes = rnd.sample(range(0, len(data[0])), len(data[0]))
    return [data[0][index] for index in indexes], [data[1][index] for index in indexes]


def split_data(data, count_image_for_train):
    x_train, x_test, y_train, y_test = [], [], [], []
    image_in_class = 10

    if count_image_for_train > 8:
        count_image_for_train = 8

    for i in range(0, len(data[0]), image_in_class):
        indexes_images_in_class = list(range(i, i + image_in_class))
        indexes_images_in_class_train = rnd.sample(indexes_images_in_class, count_image_for_train)
        x_train.extend(data[0][index] for index in indexes_images_in_class_train)
        y_train.extend(data[1][index] for index in indexes_images_in_class_train)

        indexes_images_in_class_test = rnd.sample(set(indexes_images_in_class) - set(indexes_images_in_class_train),
                                                  image_in_class - count_image_for_train)
        x_test.extend(data[0][index] for index in indexes_images_in_class_test)
        y_test.extend(data[1][index] for index in indexes_images_in_class_test)

    return x_train, x_test, y_train, y_test


def create_feature(data, method, parameter):
    method = eval(method)
    result = []
    for element in data:
        if method == get_histogram:
            result.append(method(element, parameter)[0])
        else:
            result.append(method(element, parameter))
    return result


def distance(el1, el2):
    return np.linalg.norm(np.array(el1) - np.array(el2))


def classifier(data, new_elements, method, parameter):
    featured_data = create_feature(data[0], method, parameter)
    featured_elements = create_feature(new_elements, method, parameter)
    result = []
    for element in featured_elements:
        min_el = [1000, -1]
        for i in range(len(featured_data)):
            dist = distance(element, featured_data[i])
            if dist < min_el[0]:
                min_el = [dist, i]
        if min_el[1] < 0:
            result.append(0)
        else:
            result.append(data[1][min_el[1]])
    print('res: ', result)
    return result


def test_classifier(data, test_elements, method, parameter):
    answers = classifier(data, test_elements[0], method, parameter)
    correct_answers = 0
    for i in range(len(test_elements[1])):
        if answers[i] == test_elements[1][i]:
            correct_answers += 1
    return correct_answers / len(test_elements[1])


def teach_parameter(data, test_elements, method):
    image_size = min(data[0][0].shape)
    param = (0, 0, 0)

    if method == get_histogram:
        param = (10, 300, 3)
    if method == get_dft or method == get_dct:
        param = (2, image_size, 1)
    if method == get_gradient:
        param = (2, int(data[0][0].shape[0] / 2 - 1), 1)
    if method == get_scale:
        param = (0.05, 1, 0.05)

    best_param = param[0]

    classf = test_classifier(data, test_elements, method, best_param)

    stat = [[best_param], [classf]]

    # for i in np.arange(param[0] + param[2], param[1], param[2]):
    #     new_classf = test_classifier(data, test_elements, method, i)
    #     stat[0].append(i)
    #     stat[1].append(new_classf)
    #     if new_classf > classf:
    #         classf = new_classf
    #         best_param = i

    return [best_param, classf], stat


def cross_validation(data, method, folds=3):

    if folds < 3:
        folds = 3
    per_fold = int(len(data[0]) / folds)

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    results = []
    for step in range(0, folds):

        if step == 0:
            x_train = data[0][per_fold:]
            x_test = data[0][:per_fold]
            y_train = data[1][per_fold:]
            y_test = data[1][:per_fold]
        else:
            if step == folds - 1:
                x_train = data[0][:step * per_fold]
                x_test = data[0][step * per_fold:]
                y_train = data[1][:step * per_fold]
                y_test = data[1][step * per_fold:]
            else:
                x_train = data[0][:step * per_fold] + data[0][(step + 1) * per_fold:]
                x_test = data[0][step * per_fold:(step + 1) * per_fold]
                y_train = data[1][:step * per_fold] + data[1][(step + 1) * per_fold:]
                y_test = data[1][step * per_fold:(step + 1) * per_fold]
        results.append(teach_parameter([x_train, y_train], [x_test, y_test], method))

    # res = results[0]
    # for element in results[1:]:
    #     best = element[0]
    #     stat = element[1]
    #     res[0][0] += best[0]
    #     res[0][1] += best[1]
    #     for i in range(len(stat[1])):
    #         res[1][1][i] += stat[1][i]
    # res[0][0] /= folds
    # if method != get_scale:
    #     res[0][0] = int(res[0][0])
    # res[0][1] /= folds
    # for i in range(len(res[1][1])):
    #     res[1][1][i] /= folds
    # return res


def get_all_images():
    data_faces = []
    data_target = []

    for i in range(1, 41):
        for j in range(1, 11):
            image = cv.cvtColor(cv.imread(str(PATH) + '/faces/s' + str(i) + "/" + str(j) + ".bmp"), cv.COLOR_BGR2GRAY)
            data_faces.append(image / 255)
            data_target.append(i)
    return [data_faces, data_target]


def get_histogram(image, param=30):
    hist, bins = np.histogram(image, bins=np.linspace(0, 1, param))
    return [hist, bins]


def get_dft(image, mat_side=13):
    f = np.fft.fft2(image)
    f = f[0:mat_side, 0:mat_side]
    return np.abs(f)


def get_dct(image, mat_side=13):
    c = dct(image, axis=1)
    c = dct(c, axis=0)
    c = c[0:mat_side, 0:mat_side]
    return c


def get_gradient(image, n=2):
    shape = image.shape[0]
    i, l = 0, 0
    r = n
    result = []

    while r <= shape:
        window = image[l:r, :]
        result.append(np.sum(window))
        i += 1
        l = i * n
        r = (i + 1) * n
    result = np.array(result)
    return result


def get_scale(image, scale=0.35):
    h = image.shape[0]
    w = image.shape[1]
    new_size = (int(h * scale), int(w * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def get_photo(self):
    print(str(PATH))
    number_class = self.params['number_class']
    number_photo = self.params['number_photo']
    path_to_image = str(PATH) + '/faces/s' + number_class + "/" + number_photo + ".bmp"
    pixmap = QtGui.QPixmap(path_to_image).scaled(215, 200, QtCore.Qt.KeepAspectRatio)
    self.selectedImage.setPixmap(pixmap)
    image = cv.cvtColor(cv.imread(path_to_image), cv.COLOR_BGR2GRAY)
    return image
