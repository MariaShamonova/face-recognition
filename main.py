import dataclasses
import random
from dataclasses import dataclass
from multiprocessing import Queue, Process, Pool
from itertools import repeat
from statistics import mode
import numpy as np
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

        self.get_params_for_method()
        classifier = eval(method_name)()
        face_recognizer = FaceRecognizer(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                         classifier=classifier)

        face_recognizer.teach_recognizer()
        score = face_recognizer.get_recognize_score()

        self.display_example_features(classifier)
        self.display_example_answer_algorithm(data_faces, classifier, face_recognizer)
        self.display_result_for_selected_parameter(score)
        best_param, max_score = self.display_selection_parameter(face_recognizer)

        scores = face_recognizer.cross_validation(data_faces, data_target, best_param)

        self.display_cross_validation(scores)

        score = self.parallel_computing()
        self.display_parallel_system_score(score)

        self.computed_dependence_score_by_count_images(best_param, )

    def computed_dependence_score_by_count_images(self):
        print('computing')

    def parallel_computing(self):
        num_faces_for_train = int(self.count_faces_in_train.text())
        data_faces, data_target = get_faces_data()
        x_train, y_train, x_test, y_test = faces_repository.split_data(data_faces, data_target, num_faces_for_train)

        self.get_params_for_method()

        scores_futures = []

        for method in self.methods:
            classifier = eval(method)()
            face_recognizer = FaceRecognizer(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                             classifier=classifier)
            face_recognizer.teach_recognizer()
            scores_futures.append(face_recognizer.get_answers)

        answers = self.parallelize(len(self.methods), scores_futures)

        correct_answer = 0
        for idx_test in range(len(y_test)):
            answer = mode([recognizer_answer[idx_test] for recognizer_answer in answers])
            if y_test[idx_test] == answer:
                correct_answer += 1

        return correct_answer / len(y_test)

    @staticmethod
    def parallelize(n_workers, functions):
        with Pool(n_workers) as pool:
            futures = [pool.apply_async(t) for t in functions]
            results = [fut.get() for fut in futures]
        return results

    def display_example_images_in_database(self, count_image_per_person=3):
        self.verticalLayout = QtWidgets.QVBoxLayout()
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
        self.verticalLayout_5.addLayout(self.verticalLayout)

    def display_example_features(self,  classifier):

        self.label_example_feature = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_example_feature.setObjectName("label_example_feature")
        self.label_example_feature.setText(_translate("MainWindow", "Example features for selected method: " + self.comboBox.currentText()))
        self.verticalLayout_5.addWidget(self.label_example_feature)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
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
        self.verticalLayout_5.addLayout(self.horizontalLayout_9)

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

    def display_selection_parameter(self, face_recognizer):
        self.title_selecteion_method = QtWidgets.QHBoxLayout()
        self.title_selecteion_method.setObjectName("title_selecteion_method")
        self.label_method_name = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_method_name.setStyleSheet("font-weight: 500;\n"
                                             "margin-top: 15px;")
        self.label_method_name.setObjectName("label_method_name")
        self.label_method_name.setText(_translate("MainWindow", "Selection of parameters for the best result"))
        self.title_selecteion_method.addWidget(self.label_method_name)
        self.verticalLayout_5.addLayout(self.title_selecteion_method)
        self.line = QtWidgets.QFrame(self.scrollAreaWidgetContents)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_5.addWidget(self.line)
        self.title_table_best_scors = QtWidgets.QHBoxLayout()
        self.title_table_best_scors.setObjectName("title_table_best_scors")

        # Заголовок
        self.title_table_best_scors_col_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.title_table_best_scors_col_2.setMinimumSize(QtCore.QSize(0, 30))
        self.title_table_best_scors_col_2.setObjectName("title_table_best_scors_col_2")
        self.title_table_best_scors_col_2.setText(_translate("MainWindow", "Parameter"))
        self.title_table_best_scors.addWidget(self.title_table_best_scors_col_2)
        self.title_table_best_scors_col_1 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.title_table_best_scors_col_1.setMinimumSize(QtCore.QSize(0, 30))
        self.title_table_best_scors_col_1.setObjectName("title_table_best_scors_col_1")
        self.title_table_best_scors_col_1.setText(_translate("MainWindow", "Score"))
        self.title_table_best_scors.addWidget(self.title_table_best_scors_col_1)
        self.verticalLayout_5.addLayout(self.title_table_best_scors)


        self.scrollArea_2 = QtWidgets.QScrollArea(self.scrollAreaWidgetContents)
        self.scrollArea_2.setMinimumSize(QtCore.QSize(0, 300))
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollAreaWidgetContents_3 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_3.setGeometry(QtCore.QRect(0, 0, 738, 298))
        self.scrollAreaWidgetContents_3.setObjectName("scrollAreaWidgetContents_3")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents_3)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.best_params_rows = QtWidgets.QVBoxLayout()
        self.best_params_rows.setObjectName("best_params_row")

        list_params = face_recognizer.get_list_params()
        for param, score in list_params:
            row = QtWidgets.QHBoxLayout()

            column_parameter = QtWidgets.QLabel(self.scrollAreaWidgetContents_3)
            column_parameter.setMaximumSize(QtCore.QSize(16777215, 30))
            column_parameter.setBaseSize(QtCore.QSize(0, 30))
            column_parameter.setText(_translate("MainWindow", str(param)))

            column_score = QtWidgets.QLabel(self.scrollAreaWidgetContents_3)
            column_score.setMaximumSize(QtCore.QSize(16777215, 30))
            column_score.setBaseSize(QtCore.QSize(0, 30))
            column_score.setText(_translate("MainWindow", str(score)))

            row.addWidget(column_parameter)
            row.addWidget(column_score)
            self.best_params_rows.addLayout(row)

        best_param, max_score = max(list_params, key=lambda x: x[1])

        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.best_params_rows.addItem(spacerItem1)
        self.verticalLayout_7.addLayout(self.best_params_rows)
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_3)
        self.verticalLayout_5.addWidget(self.scrollArea_2)
        self.label_of_best_result = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_of_best_result.setObjectName("label_of_best_result")
        self.verticalLayout_5.addWidget(self.label_of_best_result)
        self.label_of_best_result.setText(_translate("MainWindow", f"Best parameter: {best_param}, best score: {max_score}"))
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        return best_param, max_score

    def display_cross_validation(self, scores):
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_cross_validation = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_cross_validation.setMaximumSize(QtCore.QSize(16777215, 30))
        self.label_cross_validation.setStyleSheet("font-weight: 500;")
        self.label_cross_validation.setObjectName("label_cross_validation")
        self.label_cross_validation.setText(_translate("MainWindow", "Cross validation"))
        self.verticalLayout.addWidget(self.label_cross_validation)

        self.cross_validation_header_3 = QtWidgets.QHBoxLayout()
        self.cross_validation_header_3.setObjectName("cross_validation_header_3")
        self.cross_validation_header_fold = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.cross_validation_header_fold.setObjectName("cross_validation_header_fold")
        self.cross_validation_header_fold.setText(_translate("MainWindow", 'Amount fold'))

        self.cross_validation_header_3.addWidget(self.cross_validation_header_fold)
        self.cross_validation_header_score = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.cross_validation_header_score.setObjectName("cross_validation_header_score")
        self.cross_validation_header_score.setText(_translate("MainWindow", "Mean score"))
        self.cross_validation_header_3.addWidget(self.cross_validation_header_score)
        self.verticalLayout.addLayout(self.cross_validation_header_3)

        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        for score in scores:
            cross_validation_row = QtWidgets.QHBoxLayout()
            cross_validation_row.setObjectName("cross_validation_row")
            cross_validation_row_fold = QtWidgets.QLabel(self.scrollAreaWidgetContents)
            cross_validation_row_fold.setObjectName("cross_validation_row_fold")
            cross_validation_row_fold.setText(_translate("MainWindow", str(score[0])))
            cross_validation_row.addWidget(cross_validation_row_fold)

            cross_validation_row_score = QtWidgets.QLabel(self.scrollAreaWidgetContents)
            cross_validation_row_score.setObjectName("cross_validation_row_score")
            cross_validation_row_score.setText(_translate("MainWindow", str(score[1])))
            cross_validation_row.addWidget(cross_validation_row_score)

            self.verticalLayout_2.addLayout(cross_validation_row)

        self.verticalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout_5.addLayout(self.verticalLayout)

    # return cross-Validation_score()

    def display_parallel_system_score(self, score):
        self.verticalLayout = QtWidgets.QVBoxLayout()
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
        self.verticalLayout_5.addLayout(self.verticalLayout)

def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)  # создаем экземпляр приложения
    window = Main_Window()
    window.show()
    # window.parallel_computing()
    sys.exit(app.exec_())



if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
