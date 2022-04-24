import dataclasses
import random
import statistics
from dataclasses import dataclass
from multiprocessing import Queue, Process, Pool
from itertools import repeat
from statistics import mode
import numpy as np
from matplotlib import pyplot as plt

import faces_repository
from feature_getters import FeatureGetter, Histogram, DFT, DCT, Scale, Gradient
import pathlib
from design import Ui_MainWindow
from PyQt5 import QtWidgets, QtGui, QtCore
from faces_repository import *
from controller import FaceRecognizer
_translate = QtCore.QCoreApplication.translate

path = str(pathlib.Path(__file__).parent.resolve())


class Main_Window(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()  # Это здесь нужно для доступа к переменным, методам
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        self.data = []

        self.methods = {
            'Histogram': {'name': 'BIN', 'value': 30},
            'DFT': {'name': 'P/Q', 'value': 15},
            'DCT': {'name': 'P/Q', 'value': 15},
            'Scale': {'name': 'Scale', 'value': 0.3},
            'Gradient':  {'name': 'W', 'value': 2},
        }
        self.connect_functions()

    def connect_functions(self):
        self.resultButton.clicked.connect(self.get_result)
        self.comboBox.currentIndexChanged.connect(self.combo_box_change)
        self.param_field.textChanged[str].connect(self.on_changed)

    def on_changed(self, text):
        method_name = self.comboBox.currentText()
        self.methods[method_name]['value'] = text


    def combo_box_change(self):
        method_name = self.comboBox.currentText()

        self.param_field.setPlaceholderText(_translate("MainWindow", "Enter " + self.methods[method_name]['name']))

    def get_params_for_method(self):
        method_name = self.comboBox.currentText()

        return self.methods.get(method_name)

    def get_result(self):

        method_name = self.comboBox.currentText()

        num_faces_for_train = int(self.count_faces_in_train.text())

        data_faces, data_target = get_faces_data()

        self.display_example_images_in_database()

        x_train, y_train, x_test, y_test = faces_repository.split_data(data_faces, data_target, num_faces_for_train)


        classifier = eval(method_name)()
        face_recognizer = FaceRecognizer(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                         classifier=classifier)

        face_recognizer.teach_recognizer()
        score = face_recognizer.get_recognize_score()

        self.display_example_features(classifier)
        self.display_example_answer_algorithm(data_faces, classifier, face_recognizer)
        self.display_result_for_selected_parameter(score)

        scores_params = self.get_scores_for_selection_parameters(face_recognizer)
        best_param, max_score = max(scores_params, key=lambda x: x[1])

        scores_folds = self.get_scores_for_cross_validation_folds(face_recognizer, best_param, data_faces, data_target,)

        self.display_selection_parameters_and_scores(scores_params, scores_folds)

        scores = self.parallel_computing()
        self.display_parallel_system_score(max(scores, key=lambda x: x[1]))
        self.display_parallel_computing_result(scores)

    def build_line_plot(self, data, name):
        plt.figure(figsize=(20, 10), dpi=80)
        ax = plt.gca()
        plt.xticks(fontsize=35)
        plt.yticks(fontsize=35)
        ax.grid(linewidth=4)
        plt.plot(*zip(*data), linewidth=5)
        save_path = name + '.png'
        plt.savefig(save_path)

        return save_path

    def get_scores_for_selection_parameters(self, face_recognizer):
        return face_recognizer.get_list_params()

    def get_scores_for_cross_validation_folds(self, face_recognizer, best_param, data_faces, data_target,):
       return face_recognizer.cross_validation(data_faces, data_target, best_param)

    @staticmethod
    def parallelize(n_workers, functions):

        with Pool(n_workers) as pool:
            futures = [pool.apply_async(t) for t in functions]
            results = [fut.get() for fut in futures]

        return results

    def parallel_computing(self):
        data_faces, data_target = get_faces_data()

        folds = np.arange(2, 10)
        scores = []

        for fold in folds:
            scores_for_k_fold = []

            for x_train, y_train, x_test, y_test in split_data_for_cross_validation(data_faces, data_target, fold):
                answers_futures = []

                for method in self.methods:
                    classifier = eval(method)()
                    face_recognizer = FaceRecognizer(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                                     classifier=classifier)
                    face_recognizer.teach_recognizer()
                    answers_futures.append(face_recognizer.get_answers)

                answers = self.parallelize(len(self.methods), answers_futures)

                correct_answer = 0
                for idx_test in range(len(y_test)):
                    answer = mode([recognizer_answer[idx_test] for recognizer_answer in answers])
                    if y_test[idx_test] == answer:
                        correct_answer += 1

                score = correct_answer / len(y_test)
                scores_for_k_fold.append(score)

            scores.append((fold, statistics.mean(scores_for_k_fold)))

        return scores

    def display_parallel_computing_result(self, scores):
        label = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        label.setObjectName("label_example_feature")
        label.setText(_translate("MainWindow", "Result" ))
        self.verticalLayout_5.addWidget(label)

        horizontalLayout = QtWidgets.QHBoxLayout()
        self.verticalLayout_5.addLayout(horizontalLayout)
        horizontalLayout.setObjectName("horizontalLayout_9")

        self.example_face = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.example_feature = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.example_feature.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignLeft)

        path_to_line_chart = self.build_line_plot(scores, 'scores_folds')

        pixmap = QtGui.QPixmap(path_to_line_chart).scaled(400, 400, QtCore.Qt.KeepAspectRatio)
        self.example_face.setPixmap(pixmap)
        horizontalLayout.addWidget(self.example_face)

        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        horizontalLayout.addItem(spacerItem1)



    def display_example_images_in_database(self, count_image_per_person=3):
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.addLayout(self.verticalLayout)
        self.example_images_title = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.example_images_title.setObjectName("example_images_title")
        self.example_images_title.setText(_translate("MainWindow", "Example images in database"))
        self.verticalLayout.addWidget(self.example_images_title)
        self.example_images_wrapper = QtWidgets.QVBoxLayout()
        self.example_images_wrapper.setObjectName("example_images_wrapper")
        data_folder = path + "/faces/s"
        for i in range(1, count_image_per_person ):
            example_images_row = QtWidgets.QHBoxLayout()
            example_images_row.setObjectName("example_images_row")

            for j in range(1, count_image_per_person + 1):
                image = QtWidgets.QLabel(self.scrollAreaWidgetContents)
                image.setMinimumSize(QtCore.QSize(92, 112))
                image.setStyleSheet("background-color: rgb(200, 200, 200);")
                image.setAlignment(QtCore.Qt.AlignCenter)
                pixmap = QtGui.QPixmap(data_folder + str(i) + "/" + str(j) + ".bmp").scaled(176, 179, QtCore.Qt.KeepAspectRatio)
                image.setPixmap(pixmap)
                example_images_row.addWidget(image)

            spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            example_images_row.addItem(spacerItem1)
            self.example_images_wrapper.addLayout(example_images_row)

        self.verticalLayout.addLayout(self.example_images_wrapper)

    def display_example_features(self,  classifier):
        self.label_example_feature = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_example_feature.setObjectName("label_example_feature")
        self.label_example_feature.setText(_translate("MainWindow", "Example features for selected method: " + self.comboBox.currentText()))
        self.verticalLayout_5.addWidget(self.label_example_feature)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.verticalLayout_5.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")

        self.example_face = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.example_feature = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.example_feature.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignLeft)

        i = random.randint(1, 40)
        j = random.randint(1, 10)
        path_to_face = path + '/faces/s'+str(i)+'/'+str(j)+'.bmp'

        face = cv2.cvtColor(cv2.imread(path_to_face), cv2.COLOR_BGR2GRAY) / 255
        path_to_face_features = classifier.plot(face)

        pixmap = QtGui.QPixmap(path_to_face).scaled(92, 112, QtCore.Qt.KeepAspectRatio)
        self.example_face.setPixmap(pixmap)
        self.horizontalLayout_9.addWidget(self.example_face)


        pixmap = QtGui.QPixmap(path + '/' + path_to_face_features).scaled(400, 200, QtCore.Qt.KeepAspectRatio)
        self.example_feature.setPixmap(pixmap)
        self.horizontalLayout_9.addWidget(self.example_feature)


        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem1)

    def display_example_answer_algorithm(self, data_faces, classifier, face_recognizer):

        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_answers = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_answers.setObjectName("label_answers")
        self.label_answers.setText(_translate("MainWindow", "Example answers"))
        self.verticalLayout_2.addWidget(self.label_answers)

        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setLabelAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.formLayout.setFormAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.formLayout.setObjectName("formLayout")

        self.label_answers_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_answers_2.setObjectName("label_answers_2")
        self.label_answers_2.setText(_translate("MainWindow", "Image"))
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_answers_2)

        self.label_answers_3 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_answers_3.setObjectName("label_answers_3")
        self.label_answers_3.setText(_translate("MainWindow", "Result"))
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.label_answers_3)

        for i in range(3):
            random_i = random.choice(range(40))
            random_j = random.choice(range(10))

            random_face = data_faces[int(random_i * random_j)]
            face_feature = classifier.get_feature(random_face)
            answer, face_answer = face_recognizer.recognize_face(face_feature)

            horizontalLayout = QtWidgets.QHBoxLayout()
            horizontalLayout.setObjectName("horizontalLayout_6")

            image = QtWidgets.QLabel(self.scrollAreaWidgetContents)
            image.setMinimumSize(QtCore.QSize(92, 112))
            image.setStyleSheet("background-color: rgb(200, 200, 200);")
            image.setAlignment(QtCore.Qt.AlignCenter)
            image.setObjectName("answer_1_1")

            height, width = random_face.shape
            random_face = random_face * 255
            random_face = random_face.astype(np.uint8)

            qimage = QtGui.QImage(random_face, width, height, QtGui.QImage.Format_Grayscale8)
            pixmap = QtGui.QPixmap(qimage).scaled(92, 112, QtCore.Qt.KeepAspectRatio)
            image.setPixmap(pixmap)

            self.formLayout.setWidget(i + 2, QtWidgets.QFormLayout.FieldRole, image)

            answer = QtWidgets.QLabel(self.scrollAreaWidgetContents)
            answer.setMinimumSize(QtCore.QSize(92, 112))
            answer.setStyleSheet("background-color: rgb(200, 200, 200);")
            answer.setAlignment(QtCore.Qt.AlignCenter)
            answer.setObjectName("answer_1_2")

            height, width = face_answer.shape
            face_answer = face_answer * 255
            face_answer = face_answer.astype(np.uint8)

            qimage = QtGui.QImage(face_answer, width, height, QtGui.QImage.Format_Grayscale8)
            pixmap = QtGui.QPixmap(qimage).scaled(92, 112, QtCore.Qt.KeepAspectRatio)
            answer.setPixmap(pixmap)

            self.formLayout.setWidget(i + 2, QtWidgets.QFormLayout.LabelRole, answer)

        self.verticalLayout_2.addLayout(self.formLayout)
        self.verticalLayout_5.addLayout(self.verticalLayout_2)

    def display_result_for_selected_parameter(self, score):
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()

        self.label_result_selected_params = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        params = self.get_params_for_method()

        self.label_result_selected_params.setText(
            _translate("MainWindow", "Сlassification result for " + params.get('name') + '=' + str(params.get('value'))))
        self.horizontalLayout_10.addWidget(self.label_result_selected_params)

        self.score_for_selected_parameter = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.score_for_selected_parameter.setText(_translate("MainWindow", str(score)))
        self.horizontalLayout_10.addWidget(self.score_for_selected_parameter)

        self.verticalLayout_5.addLayout(self.horizontalLayout_10)

    def display_selection_parameters_and_scores(self, scores_params, scores_folds):
        verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.addLayout(verticalLayout)
        verticalLayout.setObjectName("verticalLayout_2")
        self.label_answers = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_answers.setObjectName("label_answers")
        self.label_answers.setText(_translate("MainWindow", "Results"))
        self.label_answers.setStyleSheet("margin-left: 15px;")
        verticalLayout.addWidget(self.label_answers)

        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setLabelAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.formLayout.setFormAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.formLayout.setObjectName("formLayout")

        label_parameters = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        label_parameters.setObjectName("label_anlabel_parametersswers_2")
        label_parameters.setText(_translate("MainWindow", "Parameters"))
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, label_parameters)

        label_folds = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        label_folds.setObjectName("label_folds")
        label_folds.setText(_translate("MainWindow", "Folds"))
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, label_folds)

        horizontalLayout = QtWidgets.QHBoxLayout()
        horizontalLayout.setObjectName("horizontalLayout_6")

        answer = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        answer.setMinimumSize(QtCore.QSize(300, 200))
        answer.setStyleSheet("background-color: rgb(200, 200, 200);")
        answer.setAlignment(QtCore.Qt.AlignCenter)
        answer.setObjectName("answer_1_2")

        # Сюда должна приходить картинка графика фолдов
        path_to_face_features = self.build_line_plot(scores_folds, 'scores_folds')
        pixmap = QtGui.QPixmap(path + '/' + path_to_face_features).scaled(400, 400, QtCore.Qt.KeepAspectRatio)
        answer.setPixmap(pixmap)

        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, answer)

        image = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        image.setMinimumSize(QtCore.QSize(300, 200))
        image.setStyleSheet("background-color: rgb(200, 200, 200);")
        image.setAlignment(QtCore.Qt.AlignCenter)
        image.setObjectName("answer_1_1")

        # Сюда должна приходить картинка графика параметров
        path_to_face_features = self.build_line_plot(scores_params, 'scores_params')
        pixmap = QtGui.QPixmap(path + '/' + path_to_face_features).scaled(400, 400, QtCore.Qt.KeepAspectRatio)
        image.setPixmap(pixmap)

        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, image)

        verticalLayout.addLayout(self.formLayout)

    def display_parallel_system_score(self, score):
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.addLayout(self.verticalLayout)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_parallel_system = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_parallel_system.setMaximumSize(QtCore.QSize(16777215, 30))
        self.label_parallel_system.setStyleSheet("")
        self.label_parallel_system.setObjectName("label_parallel_system")
        self.label_parallel_system.setText(_translate("MainWindow", "Score of parallel system"))
        self.horizontalLayout_2.addWidget(self.label_parallel_system)

        self.label_parallel_system_value = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_parallel_system_value.setMaximumSize(QtCore.QSize(16777215, 30))
        self.label_parallel_system_value.setStyleSheet("")
        self.label_parallel_system_value.setObjectName("label_parallel_system_value")
        self.label_parallel_system_value.setText(_translate("MainWindow", str(score)))
        self.horizontalLayout_2.addWidget(self.label_parallel_system_value)
        self.verticalLayout.addLayout(self.horizontalLayout_2)


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)  # создаем экземпляр приложения
    window = Main_Window()
    window.show()
    # window.parallel_computing()
    sys.exit(app.exec_())



if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
