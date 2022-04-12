import dataclasses
import numpy as np
from feature_getters import FeatureGetter


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

    def teach_recognizer(self):
        self.faces_train_featured.clear()
        self.faces_train_featured.extend(self.classifier.get_feature(face_train) for face_train in self.x_train)

    def recognize_face(self, face: np.ndarray) -> int:

        min_distance = float('inf')
        answer_idx = 0

        for idx, known_face in enumerate(self.faces_train_featured):
            distance = self._calculate_distance(known_face, face)
            if distance < min_distance:
                answer_idx = idx
                min_distance = distance

        return self.y_train[answer_idx]

    def get_recognize_score(self):
        self.faces_test_featured.extend(self.classifier.get_feature(face_test) for face_test in self.x_test)

        correct_answers = 0

        for idx_test, face_test in enumerate(self.faces_test_featured):
            right_answer = self.y_test[idx_test]
            recognizer_answer = self.recognize_face(face_test)
            if right_answer == recognizer_answer:
                correct_answers += 1

        print('Точность распознавания:', correct_answers / len(self.faces_test_featured))



# data_faces, data_target = faces_repository.get_faces_data()
# feature_getter = Histogram
#
# num_faces_for_train = 3
# print('Количество лиц для обучения:', num_faces_for_train)
#
# x_train, y_train, x_test, y_test = faces_repository.split_data(data_faces, data_target, num_faces_for_train)
# classifier = feature_getter()
# face_recognizer = FaceRecognizer(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
#                                  classifier=classifier)
#
# face_recognizer.teach_recognizer()
# face_recognizer.get_recognize_score()