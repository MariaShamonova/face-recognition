# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'recognizer.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(10, 0, 781, 531))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 764, 883))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.comboBox = QtWidgets.QComboBox(self.scrollAreaWidgetContents)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.horizontalLayout_5.addWidget(self.comboBox)
        self.stackedWidget = QtWidgets.QStackedWidget(self.scrollAreaWidgetContents)
        self.stackedWidget.setMinimumSize(QtCore.QSize(0, 50))
        self.stackedWidget.setObjectName("stackedWidget")
        self.page_5 = QtWidgets.QWidget()
        self.page_5.setObjectName("page_5")
        self.param_bin = QtWidgets.QLineEdit(self.page_5)
        self.param_bin.setGeometry(QtCore.QRect(10, 10, 115, 30))
        self.param_bin.setStyleSheet("background-color: rgb(190, 190, 190);\n"
"padding-left: 10px;")
        self.param_bin.setText("")
        self.param_bin.setObjectName("param_bin")
        self.stackedWidget.addWidget(self.page_5)
        self.page_8 = QtWidgets.QWidget()
        self.page_8.setObjectName("page_8")
        self.param_p = QtWidgets.QLineEdit(self.page_8)
        self.param_p.setGeometry(QtCore.QRect(10, 10, 115, 30))
        self.param_p.setStyleSheet("background-color: rgb(190, 190, 190);\n"
"padding-left: 10px;")
        self.param_p.setText("")
        self.param_p.setObjectName("param_p")
        self.param_q = QtWidgets.QLineEdit(self.page_8)
        self.param_q.setGeometry(QtCore.QRect(140, 10, 115, 30))
        self.param_q.setStyleSheet("background-color: rgb(190, 190, 190);\n"
"padding-left: 10px;")
        self.param_q.setText("")
        self.param_q.setObjectName("param_q")
        self.stackedWidget.addWidget(self.page_8)
        self.page_6 = QtWidgets.QWidget()
        self.page_6.setObjectName("page_6")
        self.param_pq = QtWidgets.QLineEdit(self.page_6)
        self.param_pq.setGeometry(QtCore.QRect(10, 10, 115, 30))
        self.param_pq.setStyleSheet("background-color: rgb(190, 190, 190);\n"
"padding-left: 10px;")
        self.param_pq.setText("")
        self.param_pq.setObjectName("param_pq")
        self.stackedWidget.addWidget(self.page_6)
        self.page_9 = QtWidgets.QWidget()
        self.page_9.setObjectName("page_9")
        self.param_scale = QtWidgets.QLineEdit(self.page_9)
        self.param_scale.setGeometry(QtCore.QRect(10, 10, 115, 30))
        self.param_scale.setStyleSheet("background-color: rgb(190, 190, 190);\n"
"padding-left: 10px;")
        self.param_scale.setText("")
        self.param_scale.setObjectName("param_scale")
        self.stackedWidget.addWidget(self.page_9)
        self.page_10 = QtWidgets.QWidget()
        self.page_10.setObjectName("page_10")
        self.param_w = QtWidgets.QLineEdit(self.page_10)
        self.param_w.setGeometry(QtCore.QRect(10, 10, 115, 30))
        self.param_w.setStyleSheet("background-color: rgb(190, 190, 190);\n"
"padding-left: 10px;")
        self.param_w.setText("")
        self.param_w.setClearButtonEnabled(False)
        self.param_w.setObjectName("param_w")
        self.stackedWidget.addWidget(self.page_10)
        self.horizontalLayout_5.addWidget(self.stackedWidget)
        self.verticalLayout_5.addLayout(self.horizontalLayout_5)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_count_faces = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_count_faces.setObjectName("label_count_faces")
        self.verticalLayout_4.addWidget(self.label_count_faces)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.count_faces_in_train = QtWidgets.QLineEdit(self.scrollAreaWidgetContents)
        self.count_faces_in_train.setMinimumSize(QtCore.QSize(100, 30))
        self.count_faces_in_train.setMaximumSize(QtCore.QSize(150, 16777215))
        self.count_faces_in_train.setStyleSheet("background-color: rgb(190, 190, 190);\n"
"padding-left: 10px;")
        self.count_faces_in_train.setObjectName("count_faces_in_train")
        self.horizontalLayout_4.addWidget(self.count_faces_in_train)
        self.resultButton = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.resultButton.setEnabled(True)
        self.resultButton.setMinimumSize(QtCore.QSize(120, 30))
        self.resultButton.setStyleSheet("background-color: rgb(0, 146, 14);\n"
"color: rgb(255, 255, 255);\n"
"margin-left: 15px;")
        self.resultButton.setCheckable(False)
        self.resultButton.setAutoDefault(True)
        self.resultButton.setDefault(False)
        self.resultButton.setObjectName("resultButton")
        self.horizontalLayout_4.addWidget(self.resultButton)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.verticalLayout_4.addLayout(self.horizontalLayout_4)
        self.verticalLayout_5.addLayout(self.verticalLayout_4)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_answers = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_answers.setObjectName("label_answers")
        self.verticalLayout_2.addWidget(self.label_answers)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.answer_1_1 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.answer_1_1.setMinimumSize(QtCore.QSize(92, 112))
        self.answer_1_1.setStyleSheet("background-color: rgb(200, 200, 200);")
        self.answer_1_1.setAlignment(QtCore.Qt.AlignCenter)
        self.answer_1_1.setObjectName("answer_1_1")
        self.horizontalLayout_6.addWidget(self.answer_1_1)
        self.answer_1_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.answer_1_2.setMinimumSize(QtCore.QSize(92, 112))
        self.answer_1_2.setStyleSheet("background-color: rgb(200, 200, 200);")
        self.answer_1_2.setAlignment(QtCore.Qt.AlignCenter)
        self.answer_1_2.setObjectName("answer_1_2")
        self.horizontalLayout_6.addWidget(self.answer_1_2)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem1)
        self.verticalLayout_6.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.answer_2_1 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.answer_2_1.setMinimumSize(QtCore.QSize(92, 112))
        self.answer_2_1.setStyleSheet("background-color: rgb(200, 200, 200);")
        self.answer_2_1.setAlignment(QtCore.Qt.AlignCenter)
        self.answer_2_1.setObjectName("answer_2_1")
        self.horizontalLayout_7.addWidget(self.answer_2_1)
        self.answer_2_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.answer_2_2.setMinimumSize(QtCore.QSize(92, 112))
        self.answer_2_2.setStyleSheet("background-color: rgb(200, 200, 200);")
        self.answer_2_2.setAlignment(QtCore.Qt.AlignCenter)
        self.answer_2_2.setObjectName("answer_2_2")
        self.horizontalLayout_7.addWidget(self.answer_2_2)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem2)
        self.verticalLayout_6.addLayout(self.horizontalLayout_7)
        self.verticalLayout_2.addLayout(self.verticalLayout_6)
        self.verticalLayout_5.addLayout(self.verticalLayout_2)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_result_selected_params = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_result_selected_params.setObjectName("label_result_selected_params")
        self.horizontalLayout_10.addWidget(self.label_result_selected_params)
        self.score_for_selected_parameter = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.score_for_selected_parameter.setObjectName("score_for_selected_parameter")
        self.horizontalLayout_10.addWidget(self.score_for_selected_parameter)
        self.verticalLayout_5.addLayout(self.horizontalLayout_10)
        self.title_selecteion_method = QtWidgets.QHBoxLayout()
        self.title_selecteion_method.setObjectName("title_selecteion_method")
        self.label_method_name = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_method_name.setObjectName("label_method_name")
        self.title_selecteion_method.addWidget(self.label_method_name)
        self.method_name = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.method_name.setObjectName("method_name")
        self.title_selecteion_method.addWidget(self.method_name)
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
        self.title_table_best_scors.addWidget(self.title_table_best_scors_col_2)
        self.title_table_best_scors_col_1 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.title_table_best_scors_col_1.setMinimumSize(QtCore.QSize(0, 30))
        self.title_table_best_scors_col_1.setObjectName("title_table_best_scors_col_1")
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
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.best_params_row.addItem(spacerItem3)
        self.verticalLayout_7.addLayout(self.best_params_row)
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_3)
        self.verticalLayout_5.addWidget(self.scrollArea_2)
        self.label_of_best_result = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_of_best_result.setObjectName("label_of_best_result")
        self.verticalLayout_5.addWidget(self.label_of_best_result)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(0, 0, 2, 2))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Histogram"))
        self.comboBox.setItemText(1, _translate("MainWindow", "DFT"))
        self.comboBox.setItemText(2, _translate("MainWindow", "DCT"))
        self.comboBox.setItemText(3, _translate("MainWindow", "Scale"))
        self.comboBox.setItemText(4, _translate("MainWindow", "Gradient"))
        self.param_bin.setPlaceholderText(_translate("MainWindow", "Enter BIN"))
        self.param_p.setPlaceholderText(_translate("MainWindow", "Enter P"))
        self.param_q.setPlaceholderText(_translate("MainWindow", "Enter Q"))
        self.param_pq.setPlaceholderText(_translate("MainWindow", "Enter P/Q"))
        self.param_scale.setPlaceholderText(_translate("MainWindow", "Enter Scale"))
        self.param_w.setPlaceholderText(_translate("MainWindow", "Enter W"))
        self.label_count_faces.setText(_translate("MainWindow", "Count faces in train sample "))
        self.count_faces_in_train.setText(_translate("MainWindow", "4"))
        self.count_faces_in_train.setPlaceholderText(_translate("MainWindow", "Enter № Class"))
        self.resultButton.setText(_translate("MainWindow", "Find"))
        self.label_answers.setText(_translate("MainWindow", "Example answers"))
        self.answer_1_1.setText(_translate("MainWindow", "Images 1"))
        self.answer_1_2.setText(_translate("MainWindow", "Result 1"))
        self.answer_2_1.setText(_translate("MainWindow", "Images 2"))
        self.answer_2_2.setText(_translate("MainWindow", "Result 2"))
        self.label_result_selected_params.setText(_translate("MainWindow", "Сlassification result for a selected parameter"))
        self.score_for_selected_parameter.setText(_translate("MainWindow", "Result"))
        self.label_method_name.setText(_translate("MainWindow", "Selection of parameters for the best result"))
        self.method_name.setText(_translate("MainWindow", "Method"))
        self.title_table_best_scors_col_2.setText(_translate("MainWindow", "Parameter"))
        self.title_table_best_scors_col_1.setText(_translate("MainWindow", "Score"))
        self.table_best_scors_row_1_col_1.setText(_translate("MainWindow", "TextLabel"))
        self.table_best_scors_row_1_col_2.setText(_translate("MainWindow", "TextLabel"))
        self.table_best_scors_row_2_col_1.setText(_translate("MainWindow", "TextLabel"))
        self.table_best_scors_row_2_col_2.setText(_translate("MainWindow", "TextLabel"))
        self.label_of_best_result.setText(_translate("MainWindow", "Label of best result"))
