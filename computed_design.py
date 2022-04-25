import numpy as np
import pathlib
from matplotlib import pyplot as plt

from design import Ui_MainWindow
from PyQt5 import QtWidgets, QtGui, QtCore
_translate = QtCore.QCoreApplication.translate
path = str(pathlib.Path(__file__).parent.resolve())


def create_pixmap(image):
	min_val, max_val = image.min(), image.max()
	image = 255.0 * (image - min_val) / (max_val - min_val)
	image = image.astype(np.uint8)

	height, width = image.shape
	qimage = QtGui.QImage(image, width, height, QtGui.QImage.Format_Grayscale8)
	pixmap = QtGui.QPixmap(qimage)

	return pixmap

def display_computing_algorithm(self, image, result):
	deleteItems(self.horizontalLayout_6)

	computing_image = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	computing_image.setMinimumSize(QtCore.QSize(92, 112))

	computing_result = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	computing_result.setMinimumSize(QtCore.QSize(92, 112))

	pixmap_image = create_pixmap(image)
	computing_image.setPixmap(pixmap_image)

	pixmap_result = create_pixmap(result)
	computing_result.setPixmap(pixmap_result)

	self.horizontalLayout_6.addWidget(computing_image)
	self.horizontalLayout_6.addWidget(computing_result)

def deleteItems(layout):
	if layout is not None:
		while layout.count():
			item = layout.takeAt(0)
			widget = item.widget()
			if widget is not None:
				widget.deleteLater()
			else:
				deleteItems(item.layout())

def display_parallel_system(self, validation_scores):
	remove_spacer(self.verticalLayout_8, self.spacerItem3)
	self.verticalLayout_14 = QtWidgets.QVBoxLayout()
	self.verticalLayout_14.setObjectName("verticalLayout_14")

	self.method_name_label_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents)

	self.method_name_label_2.setText(_translate("MainWindow", "Parallel system"))
	self.verticalLayout_14.addWidget(self.method_name_label_2)
	self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
	self.horizontalLayout_6.setObjectName("horizontalLayout_6")

	add_chart(self, self.horizontalLayout_6, validation_scores, "Cross Validation")

	self.verticalLayout_14.addLayout(self.horizontalLayout_6)
	self.verticalLayout_8.addLayout(self.verticalLayout_14)
	add_spacer_for_layout(self)

def add_chart(self, horizontalLayout, data, title):
	self.verticalLayout_13 = QtWidgets.QVBoxLayout()
	self.verticalLayout_13.setObjectName("verticalLayout_13")
	self.selection_param_label = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.selection_param_label.setObjectName("selection_param_label")
	self.selection_param_label.setText(_translate("MainWindow", title))
	self.verticalLayout_13.addWidget(self.selection_param_label)

	chart = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	chart.setMinimumSize(QtCore.QSize(0, 200))

	path_to_face_features = build_line_plot(data, 'scores_folds')
	pixmap = QtGui.QPixmap(path + '/' + path_to_face_features).scaled(350, 400, QtCore.Qt.KeepAspectRatio)
	chart.setPixmap(pixmap)

	self.verticalLayout_13.addWidget(chart)
	horizontalLayout.addLayout(self.verticalLayout_13)

def display_line(self):
	self.line = QtWidgets.QFrame(self.scrollAreaWidgetContents)
	self.line.setFrameShape(QtWidgets.QFrame.HLine)
	self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
	self.line.setObjectName("line")
	self.verticalLayout_8.addWidget(self.line)

def display_scores(self, scores_params, scores_folds, method_name, param_name, best_param, max_score):
	self.verticalLayout_10 = QtWidgets.QVBoxLayout()
	self.verticalLayout_10.setObjectName("verticalLayout_10")
	self.method_name_label = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.method_name_label.setObjectName("method_name_label")
	self.method_name_label.setText(_translate("MainWindow", method_name))
	self.verticalLayout_10.addWidget(self.method_name_label)

	self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
	self.horizontalLayout_5.setObjectName("horizontalLayout_5")

	add_chart(self,self.horizontalLayout_5, scores_params, "Selection Parameter")
	add_chart(self,self.horizontalLayout_5, scores_folds, "Cross Validation")


	self.verticalLayout_10.addLayout(self.horizontalLayout_5)

	self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
	self.horizontalLayout_11.setObjectName("horizontalLayout_11")
	self.label_best_parm = QtWidgets.QLabel(self.scrollAreaWidgetContents)
	self.label_best_parm.setObjectName("label_best_parm")
	self.label_best_parm.setText(_translate("MainWindow", f"Best parameter is {param_name}={best_param}. Best score is {max_score}"))
	self.horizontalLayout_11.addWidget(self.label_best_parm)
	spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
	self.horizontalLayout_11.addItem(spacerItem3)
	self.verticalLayout_10.addLayout(self.horizontalLayout_11)
	self.verticalLayout_8.addLayout(self.verticalLayout_10)


def build_line_plot(data, name):
	plt.figure(figsize=(20, 10), dpi=80)
	ax = plt.gca()
	ax.grid(linewidth=6)
	plt.xticks(fontsize=55)
	plt.yticks(fontsize=55)
	plt.rcParams["font.weight"] = 500

	plt.plot(*zip(*data), linewidth=6)
	save_path = name + '.png'
	plt.savefig(save_path)

	return save_path

def remove_spacer(layout, spacer):
	layout.removeItem(spacer)
def add_spacer_for_layout(self):
	self.spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
	self.verticalLayout_8.addItem(self.spacerItem3)

def add_spacer(lauout, column, row):
	spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
	lauout.addItem(spacerItem, row, column, 1, 1)

def add_feature(self, path_image,  name, column, row):
	verticalLayout = QtWidgets.QVBoxLayout()
	label = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
	label.setText(_translate("MainWindow", name))
	verticalLayout.addWidget(label)

	feature = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)

	if name == 'Histogram' or name == 'Gradient':
		pixmap = QtGui.QPixmap(path_image).scaled(250, 250, QtCore.Qt.KeepAspectRatio)
	elif name == 'Scale':
		pixmap = QtGui.QPixmap(path_image).scaled(155, 250, QtCore.Qt.KeepAspectRatio)

	elif name == 'DCT' or name == 'DFT':
		pixmap = QtGui.QPixmap(path_image).scaled(250, 350, QtCore.Qt.KeepAspectRatio)
	else:
		pixmap = QtGui.QPixmap(path_image)

	feature.setPixmap(pixmap)


	verticalLayout.addWidget(feature)
	self.gridLayout_3.addLayout(verticalLayout, row, column, 1, 1)

def add_result(self, image, name, column, row):
	min_val, max_val = image.min(), image.max()
	image = 255.0 * (image - min_val) / (max_val - min_val)
	image = image.astype(np.uint8)

	height, width = image.shape

	qimage = QtGui.QImage(image, width, height, QtGui.QImage.Format_Grayscale8)

	verticalLayout = QtWidgets.QVBoxLayout()
	label = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
	label.setText(_translate("MainWindow", name))
	verticalLayout.addWidget(label)

	feature = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)

	pixmap = QtGui.QPixmap(qimage)

	feature.setPixmap(pixmap)

	verticalLayout.addWidget(feature)
	self.gridLayout_7.addLayout(verticalLayout, row, column, 1, 1)

def enable_button_run_recognizer(self):
	self.buttonRunRecognizer.setEnabled(True)
	self.buttonRunRecognizer.setStyleSheet("color: rgb(0,0,0);\n"
										   "background-color: rgb(0, 114, 0, 112);"
										   "padding: 0 20px;\n"
										   "border-radius: 4px;\n"
										   )

def disable_button_run_recognizer(self):
	self.buttonRunRecognizer.setEnabled(False)
	self.buttonRunRecognizer.setStyleSheet("color: rgb(142, 142, 142);\n"		
										   "background-color: rgb(198, 198, 198);"
										   "padding: 0 20px;\n"
										   "border-radius: 4px;\n"
										   )

def enable_buttons_explore(self):
	self.buttonExploreMethods.setEnabled(True)
	self.buttonParallelSystem.setEnabled(True)
	self.buttonExploreMethods.setStyleSheet("color: rgb(0,0,0);\n"
										   "background-color: rgb(0, 114, 0, 112);"
										   "padding: 0 20px;\n"
										   "border-radius: 4px;\n"
										   )
	self.buttonParallelSystem.setStyleSheet("color: rgb(0,0,0);\n"
											"background-color: rgb(0, 57, 127, 81);"
											"padding: 0 20px;\n"
											"border-radius: 4px;\n"
											)