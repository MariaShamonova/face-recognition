import dataclasses
import random
from dataclasses import dataclass

import numpy as np
import faces_repository
from feature_getters import FeatureGetter, Histogram, DFT, DCT, Scale
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
            'Histogram': {'name': 'bin', 'value': 30},
            'DFT': {'name': 'p', 'value': 15},
            'DCT': {'name': 'pq', 'value': 15},
            'Scale': {'name': 'scale', 'value': 0.3},
            'Gradient':  {'name': 'w', 'value': 0},
        }
        self.connect_functions()

    def connect_functions(self):
        self.resultButton.clicked.connect(self.get_result)
        self.comboBox.currentIndexChanged.connect(self.combo_box_change)
        self.param_bin.textChanged[str].connect(lambda e: self.on_changed(e, 'bin'))
        self.param_p.textChanged[str].connect(lambda e: self.on_changed(e, 'P'))
        self.param_pq.textChanged[str].connect(lambda e: self.on_changed(e, 'PQ'))
        self.param_scale.textChanged[str].connect(lambda e: self.on_changed(e, 'scale'))

    def on_changed(self, text, name):
        method_name = self.comboBox.currentText()
        self.methods[method_name]['value'] = text

    def combo_box_change(self):
        item_id = self.comboBox.currentIndex()

        self.stackedWidget.setCurrentIndex(item_id)

    def get_params_for_method(self):
        method_name = self.comboBox.currentText()

        return self.methods.get(method_name)

    def get_result(self):
        print(self.comboBox.currentText())
        feature_getter = eval(self.comboBox.currentText())
        num_faces_for_train = int(self.count_faces_in_train.text())

        data_faces, data_target = get_faces_data()
        self.display_example_images_in_database()

        x_train, y_train, x_test, y_test = faces_repository.split_data(data_faces, data_target, num_faces_for_train)

        self.get_params(self.comboBox.currentText())
        classifier = feature_getter()
        face_recognizer = FaceRecognizer(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                         classifier=classifier)

        face_recognizer.teach_recognizer()
        score = face_recognizer.get_recognize_score()
        self.display_example_features(classifier)

        self.display_example_answer_algorithm(data_faces, classifier, face_recognizer)

        print(score)
        self.display_result_for_selected_parameter(score)

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
        print(params)
        self.label_result_selected_params.setText(
            _translate("MainWindow", "Сlassification result for " + params.get('name') + '=' + str(params.get('value'))))
        self.horizontalLayout_10.addWidget(self.label_result_selected_params)

        self.score_for_selected_parameter = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.score_for_selected_parameter.setText(_translate("MainWindow", str(score)))
        self.horizontalLayout_10.addWidget(self.score_for_selected_parameter)

        self.verticalLayout_5.addLayout(self.horizontalLayout_10)

    def display_selection_parameter(self):


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
        self.best_params_row = QtWidgets.QVBoxLayout()
        self.best_params_row.setObjectName("best_params_row")
        self.best_params_1 = QtWidgets.QHBoxLayout()
        self.best_params_1.setObjectName("best_params_1")
        self.table_best_scors_row_1_col_1 = QtWidgets.QLabel(self.scrollAreaWidgetContents_3)
        self.table_best_scors_row_1_col_1.setMinimumSize(QtCore.QSize(0, 30))
        self.table_best_scors_row_1_col_1.setMaximumSize(QtCore.QSize(16777215, 30))
        self.table_best_scors_row_1_col_1.setBaseSize(QtCore.QSize(0, 30))
        self.table_best_scors_row_1_col_1.setObjectName("table_best_scors_row_1_col_1")
        self.best_params_1.addWidget(self.table_best_scors_row_1_col_1)
        self.table_best_scors_row_1_col_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents_3)
        self.table_best_scors_row_1_col_2.setMaximumSize(QtCore.QSize(16777215, 30))
        self.table_best_scors_row_1_col_2.setBaseSize(QtCore.QSize(0, 30))
        self.table_best_scors_row_1_col_2.setObjectName("table_best_scors_row_1_col_2")
        self.best_params_1.addWidget(self.table_best_scors_row_1_col_2)
        self.best_params_row.addLayout(self.best_params_1)
        self.best_params_2 = QtWidgets.QHBoxLayout()
        self.best_params_2.setObjectName("best_params_2")
        self.table_best_scors_row_2_col_1 = QtWidgets.QLabel(self.scrollAreaWidgetContents_3)
        self.table_best_scors_row_2_col_1.setMinimumSize(QtCore.QSize(0, 30))
        self.table_best_scors_row_2_col_1.setMaximumSize(QtCore.QSize(16777215, 30))
        self.table_best_scors_row_2_col_1.setObjectName("table_best_scors_row_2_col_1")
        self.best_params_2.addWidget(self.table_best_scors_row_2_col_1)
        self.table_best_scors_row_2_col_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents_3)
        self.table_best_scors_row_2_col_2.setMinimumSize(QtCore.QSize(0, 30))
        self.table_best_scors_row_2_col_2.setMaximumSize(QtCore.QSize(16777215, 30))
        self.table_best_scors_row_2_col_2.setObjectName("table_best_scors_row_2_col_2")
        self.best_params_2.addWidget(self.table_best_scors_row_2_col_2)
        self.best_params_row.addLayout(self.best_params_2)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.best_params_row.addItem(spacerItem1)
        self.verticalLayout_7.addLayout(self.best_params_row)
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_3)
        self.verticalLayout_5.addWidget(self.scrollArea_2)
        self.label_of_best_result = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_of_best_result.setObjectName("label_of_best_result")
        self.verticalLayout_5.addWidget(self.label_of_best_result)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.label_of_best_result.setText(_translate("MainWindow", "Label of best result"))

def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)  # создаем экземпляр приложения
    window = Main_Window()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
