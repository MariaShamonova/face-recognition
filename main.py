import cv2 as cv
import os
from pathlib import Path
from design import Ui_MainWindow
from PyQt5 import QtWidgets, QtGui, QtCore
from functions import *
import numpy as np
from sklearn.model_selection import train_test_split

PATH = Path(__file__).parent.absolute()


class Main_Window(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()  # Это здесь нужно для доступа к переменным, методам
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        self.add_functions()

        self.data = []
        self.count_image_for_train = 4


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


    def add_functions(self):
        self.resultButton.clicked.connect(self.get_result)
        self.comboBox.currentIndexChanged.connect(self.combo_box_change)

        self.param_bin.textChanged[str].connect(lambda e: self.on_changed(e, 'bin'))
        self.param_p.textChanged[str].connect(lambda e: self.on_changed(e, 'P'))
        self.param_q.textChanged[str].connect(lambda e: self.on_changed(e, 'Q'))
        self.param_pq.textChanged[str].connect(lambda e: self.on_changed(e, 'PQ'))
        self.param_scale.textChanged[str].connect(lambda e: self.on_changed(e, 'scale'))

        self.param_class.textChanged[str].connect(lambda e: self.on_changed(e, 'number_class'))
        self.param_photo.textChanged[str].connect(lambda e: self.on_changed(e, 'number_photo'))

    def on_changed(self, text, name):
        self.params[name] = text

    def combo_box_change(self):
        item_id = self.comboBox.currentIndex()
        self.stackedWidget.setCurrentIndex(item_id)

    def get_result(self):
        self.data = get_all_images()
        self.count_image_for_train = 4

        x_train, x_test, y_train, y_test = split_data(self.data, self.count_image_for_train)

        train = mesh_data([x_train, y_train])
        test = mesh_data([x_test, y_test])

        for f in [3, 5, 7]:
            res = cross_validation(train, self.methods[self.comboBox.currentIndex()], folds=f)




def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)  # создаем экземпляр приложения
    window = Main_Window()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
