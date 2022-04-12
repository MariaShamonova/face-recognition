from design import Ui_MainWindow
from PyQt5 import QtWidgets, QtGui, QtCore

_translate = QtCore.QCoreApplication.translate


def display_example_images_in_database(self, images, count_image_per_person=3):

	# Создаем вертикальный layout
	# self.verticalLayout = QtWidgets.QVBoxLayout()
	self.verticalLayout.setObjectName("verticalLayout")
	self.example_images_title = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.example_images_title.setObjectName("example_images_title")
	self.example_images_title.setText(_translate("MainWindow", "Example images in database"))
	self.verticalLayout.addWidget(self.example_images_title)

	self.example_images_wrapper = QtWidgets.QVBoxLayout()
	self.example_images_wrapper.setObjectName("example_images_wrapper")
	print(self['example_1_1'])

	# Создаем первое изображение
	for i in range(0, images, count_image_per_person):
		self.example_images_row_1 = QtWidgets.QHBoxLayout()
		self.example_images_row_1.setObjectName("example_images_row_1")

		for j in range(count_image_per_person):
			self.example_1_1 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
			self.example_1_1.setMinimumSize(QtCore.QSize(92, 112))
			self.example_1_1.setStyleSheet("background-color: rgb(200, 200, 200);")
			self.example_1_1.setAlignment(QtCore.Qt.AlignCenter)
			self.example_1_1.setObjectName("example_1_1")
			self.example_images_row_1.addWidget(self.example_1_1)

	self.example_1_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.example_1_2.setMinimumSize(QtCore.QSize(92, 112))
	self.example_1_2.setStyleSheet("background-color: rgb(200, 200, 200);")
	self.example_1_2.setAlignment(QtCore.Qt.AlignCenter)
	self.example_1_2.setObjectName("example_1_2")
	self.example_images_row_1.addWidget(self.example_1_2)

	self.example_1_3 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.example_1_3.setMinimumSize(QtCore.QSize(92, 112))
	self.example_1_3.setStyleSheet("background-color: rgb(200, 200, 200);")
	self.example_1_3.setAlignment(QtCore.Qt.AlignCenter)
	self.example_1_3.setObjectName("example_1_3")
	self.example_images_row_1.addWidget(self.example_1_3)

	spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
	self.example_images_row_1.addItem(spacerItem1)
	self.example_images_wrapper.addLayout(self.example_images_row_1)

	# Создаем вторую строку
	self.example_images_row_2 = QtWidgets.QHBoxLayout()
	self.example_images_row_2.setObjectName("example_images_row_2")
	self.example_2_1 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.example_2_1.setMinimumSize(QtCore.QSize(92, 112))
	self.example_2_1.setStyleSheet("background-color: rgb(200, 200, 200);")
	self.example_2_1.setAlignment(QtCore.Qt.AlignCenter)
	self.example_2_1.setObjectName("example_2_1")
	self.example_images_row_2.addWidget(self.example_2_1)
	self.example_2_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.example_2_2.setMinimumSize(QtCore.QSize(92, 112))
	self.example_2_2.setStyleSheet("background-color: rgb(200, 200, 200);")
	self.example_2_2.setAlignment(QtCore.Qt.AlignCenter)
	self.example_2_2.setObjectName("example_2_2")
	self.example_images_row_2.addWidget(self.example_2_2)
	self.example_2_3 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.example_2_3.setMinimumSize(QtCore.QSize(92, 112))
	self.example_2_3.setStyleSheet("background-color: rgb(200, 200, 200);")
	self.example_2_3.setAlignment(QtCore.Qt.AlignCenter)
	self.example_2_3.setObjectName("example_2_3")
	self.example_images_row_2.addWidget(self.example_2_3)
	spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
	self.example_images_row_2.addItem(spacerItem2)
	self.example_images_wrapper.addLayout(self.example_images_row_2)
	self.verticalLayout.addLayout(self.example_images_wrapper)
	self.verticalLayout_5.addLayout(self.verticalLayout)
