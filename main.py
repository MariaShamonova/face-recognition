import dataclasses
import multiprocessing
import os
from time import sleep
import random
import statistics
from dataclasses import dataclass
from multiprocessing import Queue, Process, Pool
from itertools import repeat
from statistics import mode
import numpy as np
from matplotlib import pyplot as plt
from computed_design import *
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

        self.selected_faces_index = 0

        self.data_faces = []
        self.data_target = []

        self.data_deidentify_faces = []
        self.data_deidentify_target = []

        self.data_with_mask_faces = []
        self.data_with_mask_target = []

        self.checkBox_image_download.setVisible(False)
        self.checkBox_images_download.setVisible(False)
        self.label_extraction_feature.setVisible(False)
        self.label_result.setVisible(False)

        self.buttonRunRecognizer.setEnabled(False)
        self.buttonExploreMethods.setEnabled(False)
        self.buttonParallelSystem.setEnabled(False)
      

        self.path_to_file = ''
        self.selected_face = ''

        self.param_histogram.setText('16')
        self.param_dft.setText('12')
        self.param_dct.setText('11')
        self.param_scale.setText('0.3')
        self.param_gradient.setText('8')

        self.methods = {
            Histogram: {'name': 'BIN', 'input_field': self.param_histogram},
            DFT: {'name': 'P/Q', 'input_field': self.param_dft},
            DCT: {'name': 'P/Q', 'input_field': self.param_dct},
            Scale: {'name': 'Scale', 'input_field': self.param_scale},
            Gradient:  {'name': 'W', 'input_field': self.param_gradient},
        }
        self.connect_functions()

    def connect_functions(self):
        self.buttonUploadDatabases.clicked.connect(self.upload_databases)
        self.buttonSelectFace.clicked.connect(self.select_face_from_browser)
        self.buttonRunRecognizer.clicked.connect(self.run_recognizer)
        self.buttonExploreMethods.clicked.connect(self.explore_methods)
        self.buttonParallelSystem.clicked.connect(self.explore_parallel_system)

    def upload_databases(self):
        self.selected_faces_index = self.comboBox.currentIndex()
        if self.selected_faces_index == 1:
            self.data_deidentify_faces, self.data_deidentify_target = get_faces_data('/fawkes', '.jpg')
        elif self.selected_faces_index == 2:
            # generate_mask_on_images(40, 10)
            self.data_with_mask_faces, self.data_with_mask_target = get_faces_data('/masks', '.png')

        self.data_faces, self.data_target = get_faces_data('/faces', '.bmp')
        self.checkBox_images_download.setVisible(True)
        enable_buttons_explore(self)

    def select_face_from_browser(self):
        self.path_to_file = self.browserFiles()
        self.selected_face = cv2.cvtColor(cv2.imread(self.path_to_file), cv2.COLOR_BGR2GRAY) / 255
        self.checkBox_image_download.setVisible(True)
        enable_button_run_recognizer(self)

        add_feature(self,self.path_to_file, 'Query Image', 0, 0)


    def run_recognizer(self):
        self.label_extraction_feature.setVisible(True)
        self.label_result.setVisible(True)

        add_result(self, self.selected_face, 'Query Image', 0, 0)

        self.data_faces, self.data_target = get_faces_data('/faces')

        column = 1
        row = 0

        for idx, method in enumerate(self.methods):

            if idx == 2:
                add_spacer(self.gridLayout_3, 3, 0)
                add_spacer(self.gridLayout_7, 3, 0)
                column = 0
                row = 1

            if idx == 4:
                add_spacer(self.gridLayout_3, 3, 1)
                add_spacer(self.gridLayout_7, 3, 1)

            face_recognizer, classifier = self.computed_features(method)
            feature_selected_image = classifier.get_feature(self.selected_face)

            path_image = classifier.plot(self.selected_face)
            add_feature(self, path_image, method.__name__, column, row)


            image = face_recognizer.recognize_face(feature_selected_image)[1]
            add_result(self, image, method.__name__,  column, row)

            column += 1

    def explore_methods(self):

        for idx, method in enumerate(self.methods):
            face_recognizer, classifier = self.computed_features(
                method,
                4,
                self.selected_faces_index
            )

            scores_params = face_recognizer.get_list_params()
            best_param, max_score = max(scores_params, key=lambda x: x[1])
            scores_folds = face_recognizer.cross_validation(self.data_faces, self.data_target, best_param)

            display_scores(self, scores_params, scores_folds, method.__name__, self.methods[method].get('name'), best_param, max_score)

            if idx != len(self.methods):
                display_line(self)

    @staticmethod
    def parallelize(n_workers, functions):

        with Pool(n_workers) as pool:
            futures = [pool.apply_async(t) for t in functions]
            results = [fut.get() for fut in futures]

        return results

    def explore_parallel_system(self):
        data_faces, data_target = get_faces_data()
        folds = np.arange(2, 10)
        scores_validation = []
        # self.computing_image.show()
        # self.computing_result.show()
        # self.label_computing.show()

        for fold in folds:
            # self.label_computing.setText(_translate("MainWindow", f"Computing on {fold} folds"))
            scores_for_k_fold = []

            for x_train, y_train, x_test, y_test in split_data_for_cross_validation(data_faces, data_target, fold):

                answers_futures = []

                for method in self.methods:
                    classifier = method()
                    face_recognizer = FaceRecognizer(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                                     classifier=classifier)
                    face_recognizer.teach_recognizer()
                    answers_futures.append(face_recognizer.get_answers)

                answers = self.parallelize(len(self.methods), answers_futures)

                correct_answer = 0

                for idx_test in range(len(y_test)):
                    answer = mode([recognizer_answer[idx_test] for recognizer_answer in answers])

                    # display_computing_algorithm(self, x_test[idx_test], data_faces[(answer - 1) * 10])

                    if y_test[idx_test] == answer:
                        correct_answer += 1

                score = correct_answer / len(y_test)
                scores_for_k_fold.append(score)

            scores_validation.append((fold, statistics.mean(scores_for_k_fold)))

        display_parallel_system(self, scores_validation)

    def on_changed(self, text):
        method_name = self.comboBox.currentText()
        self.methods[method_name]['value'] = text

    def browserFiles(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Image',
                                                      str(path) + '/faces')
        return fname[0]

    def computed_features(self, method, num_faces_for_train: int = 4, selected_faces_index: bool = 0):

        x_train, y_train, x_test, y_test = faces_repository.split_data(
            self.data_faces,
            self.data_target,
            self.data_deidentify_faces,
            self.data_with_mask_faces,
            num_faces_for_train,
            selected_faces_index,
        )
        value = self.methods[method]['input_field'].text()

        if method.__name__ == 'Scale':
            param = float(value)
        else:
            param = int(value)

        classifier = method(param)
        face_recognizer = FaceRecognizer(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                         classifier=classifier)

        face_recognizer.teach_recognizer()
        return face_recognizer, classifier

    def pyt_on_mask(self):
        print('pyt_on_mask')

    def deidentify_faces(self):
        print('deidentify_faces')

def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)  # создаем экземпляр приложения
    window = Main_Window()
    window.show()
    # window.parallel_computing()
    sys.exit(app.exec_())



if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
