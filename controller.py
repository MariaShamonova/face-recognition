import dataclasses
import random
import statistics

import cv2
import numpy as np
from feature_getters import FeatureGetter
from faces_repository import split_data_for_cross_validation

@dataclasses.dataclass
class FaceRecognizer:
    x_train: list
    y_train: list

    x_test: list
    y_test: list

    classifier: FeatureGetter

    faces_train_featured: list = dataclasses.field(default_factory=list)
    faces_test_featured: list = dataclasses.field(default_factory=list)

    @staticmethod
    def _calculate_distance(array_1: np.ndarray, array_2: np.ndarray) -> float:
        return np.linalg.norm(np.array(array_1) - np.array(array_2))

    def set_new_data(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_list_params(self):
        list_params = []
        for param in self.classifier.get_teach_param(self.x_train[0]):
            self.update_features(param)
            list_params.append((param, self.get_recognize_score()))

        return list_params

    def update_features(self, param: int):
        self.classifier.set_param(param)
        self.teach_recognizer()

    def teach_recognizer(self):
        self.faces_train_featured.clear()
        self.faces_test_featured.clear()

        self.faces_train_featured.extend(self.classifier.get_feature(face_train) for face_train in self.x_train)
        self.faces_test_featured.extend(self.classifier.get_feature(face_test) for face_test in self.x_test)

    def get_answers(self):
        answers = []

        for idx_test, face_test in enumerate(self.faces_test_featured):
            answers.append(self.recognize_face(face_test)[0])

        return answers

    def recognize_face(self, face: np.ndarray) -> tuple[int, np.ndarray]:
        min_distance = float('inf')
        answer_idx = 0

        for idx, known_face in enumerate(self.faces_train_featured):
            distance = self._calculate_distance(known_face, face)
            if distance < min_distance:
                answer_idx = idx
                min_distance = distance

        return self.y_train[answer_idx], self.x_train[answer_idx]


    def get_recognize_score(self):

        correct_answers = 0

        for idx_test, face_test in enumerate(self.faces_test_featured):
            right_answer = self.y_test[idx_test]
            recognizer_answer = self.recognize_face(face_test)[0]

            if right_answer == recognizer_answer:
                correct_answers += 1

        return correct_answers / len(self.faces_test_featured)

    def cross_validation(self, train_data: np.ndarray, target_data: np.ndarray, param: int):
        folds = np.arange(2, 10)
        scores = []
        for fold in folds:
            scores_for_k_fold = []

            for x_train, y_train, x_test, y_test in split_data_for_cross_validation(train_data, target_data, fold):
                self.set_new_data( x_train, y_train, x_test, y_test)

                self.update_features(param)
                self.teach_recognizer()

                scores_for_k_fold.append(self.get_recognize_score())

            scores.append((fold, statistics.mean(scores_for_k_fold)))

        return scores

