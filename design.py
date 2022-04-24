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
        MainWindow.resize(826, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(10, 10, 791, 531))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 789, 529))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.comboBox = QtWidgets.QComboBox(self.scrollAreaWidgetContents)
        self.comboBox.setMaximumSize(QtCore.QSize(150, 30))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.horizontalLayout_5.addWidget(self.comboBox)
        self.param_field = QtWidgets.QLineEdit(self.scrollAreaWidgetContents)
        self.param_field.setMinimumSize(QtCore.QSize(0, 30))
        self.param_field.setMaximumSize(QtCore.QSize(150, 30))
        self.param_field.setStyleSheet("background-color: rgb(190, 190, 190);\n"
"padding-left: 10px;")
        self.param_field.setText("")
        self.param_field.setObjectName("param_field")
        self.horizontalLayout_5.addWidget(self.param_field)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem)
        self.verticalLayout_5.addLayout(self.horizontalLayout_5)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_count_faces = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_count_faces.setMaximumSize(QtCore.QSize(16777215, 30))
        self.label_count_faces.setObjectName("label_count_faces")
        self.verticalLayout_4.addWidget(self.label_count_faces)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
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
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem1)
        self.verticalLayout_4.addLayout(self.horizontalLayout_4)
        self.verticalLayout_5.addLayout(self.verticalLayout_4)
        self.title_selecteion_method = QtWidgets.QHBoxLayout()
        self.title_selecteion_method.setObjectName("title_selecteion_method")
        self.verticalLayout_5.addLayout(self.title_selecteion_method)
        self.line = QtWidgets.QFrame(self.scrollAreaWidgetContents)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_5.addWidget(self.line)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem2)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(0, 0, 2, 2))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 826, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Histogram"))
        self.comboBox.setItemText(1, _translate("MainWindow", "DFT"))
        self.comboBox.setItemText(2, _translate("MainWindow", "DCT"))
        self.comboBox.setItemText(3, _translate("MainWindow", "Scale"))
        self.comboBox.setItemText(4, _translate("MainWindow", "Gradient"))
        self.param_field.setPlaceholderText(_translate("MainWindow", "Enter BIN"))
        self.label_count_faces.setText(_translate("MainWindow", "Count faces in train sample "))
        self.count_faces_in_train.setText(_translate("MainWindow", "4"))
        self.count_faces_in_train.setPlaceholderText(_translate("MainWindow", "Enter № Class"))
        self.resultButton.setText(_translate("MainWindow", "Find"))
