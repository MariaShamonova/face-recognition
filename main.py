import dataclasses
from dataclasses import dataclass

import numpy as np
import faces_repository
from feature_getters import FeatureGetter, Histogram, DFT, DCT
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
        self.params = {
            'bin': '',
            'P': '',
            'Q': '',
            'PQ': '',
            'w': '',
            'scale': '',
            'number_class': '1',
            'number_photo': '1'
        }
        self.methods = ('get_histogram', 'get_dft', 'get_dct', 'get_scale', 'get_gradient')
        self.connect_functions()

    def connect_functions(self):
        self.resultButton.clicked.connect(self.get_result)
        self.comboBox.currentIndexChanged.connect(self.combo_box_change)
        self.param_bin.textChanged[str].connect(lambda e: self.on_changed(e, 'bin'))
        self.param_p.textChanged[str].connect(lambda e: self.on_changed(e, 'P'))
        self.param_q.textChanged[str].connect(lambda e: self.on_changed(e, 'Q'))
        self.param_pq.textChanged[str].connect(lambda e: self.on_changed(e, 'PQ'))
        self.param_scale.textChanged[str].connect(lambda e: self.on_changed(e, 'scale'))

    def on_changed(self, text, name):
        self.params[name] = text

    def combo_box_change(self):
        item_id = self.comboBox.currentIndex()
        self.stackedWidget.setCurrentIndex(item_id)

    def get_result(self):
        feature_getter = eval(self.comboBox.currentText())
        num_faces_for_train = int(self.count_faces_in_train.text())

        data_faces, data_target = get_faces_data()
        self.display_example_images_in_database()

        x_train, y_train, x_test, y_test = faces_repository.split_data(data_faces, data_target, num_faces_for_train)
        classifier = feature_getter()
        face_recognizer = FaceRecognizer(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                         classifier=classifier)

        face_recognizer.teach_recognizer()
        self.display_example_features(data_faces[0], classifier)

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

    def display_example_features(self, face_features, classifier):

        self.label_example_feature = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_example_feature.setObjectName("label_example_feature")
        self.label_example_feature.setText(_translate("MainWindow", "Example features for selected method: " + self.comboBox.currentText()))
        self.verticalLayout_5.addWidget(self.label_example_feature)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.example_feature = QtWidgets.QLabel(self.scrollAreaWidgetContents)


        self.example_feature.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.example_feature.setObjectName("example_feature")

        result_folder = classifier.plot(face_features)
        print(result_folder)
        pixmap = QtGui.QPixmap(path + '/' + result_folder).scaled(400, 200, QtCore.Qt.KeepAspectRatio)
        self.example_feature.setPixmap(pixmap)

        self.horizontalLayout_9.addWidget(self.example_feature)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem1)
        self.verticalLayout_5.addLayout(self.horizontalLayout_9)

def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)  # создаем экземпляр приложения
    window = Main_Window()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
