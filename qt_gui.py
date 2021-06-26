# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap, QPainter


class ui_main_window(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 741)
        MainWindow.setMinimumSize(QtCore.QSize(1152, 648))
        MainWindow.setMaximumSize(QtCore.QSize(1152, 648))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(150, 10, 1200, 600))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setContentsMargins(-1, -1, 0, -1)
        self.verticalLayout_5.setObjectName("verticalLayout_5")

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.horizontalLayout_2.setContentsMargins(-1, 0, 0, 40)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(0, -1, -1, 0)
        self.verticalLayout.setSpacing(20)
        self.verticalLayout.setObjectName("verticalLayout")

        self.horizontalLayout_2.addLayout(self.verticalLayout)

        self.button_gesture_detection = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.button_gesture_detection .setObjectName("button_gesture_detection ")
        self.horizontalLayout_2.addWidget(self.button_gesture_detection)

        self.button_gesture_recognition = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.button_gesture_recognition.setObjectName("button_gesture_recognition")
        self.horizontalLayout_2.addWidget(self.button_gesture_recognition)

        self.button_gesture_score = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.button_gesture_score.setObjectName("button_gesture_score")
        self.horizontalLayout_2.addWidget(self.button_gesture_score)

        self.verticalLayout_5.addLayout(self.horizontalLayout_2)

        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_5.setMinimumSize(QtCore.QSize(960, 544))
        self.label_5.setMaximumSize(QtCore.QSize(960, 544))
        self.label_5.setObjectName("label_5")

        self.verticalLayout_5.addWidget(self.label_5)
        self.horizontalLayout.addLayout(self.verticalLayout_5)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)

        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")

        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")

        self.verticalLayout_3.addLayout(self.horizontalLayout_7)

        self.verticalLayout_2.addLayout(self.verticalLayout_3)

        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")

        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_6.setMinimumSize(QtCore.QSize(200, 247))
        self.label_6.setMaximumSize(QtCore.QSize(200, 247))
        self.label_6.setObjectName("label_6")

        self.horizontalLayout_4.addWidget(self.label_6)

        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label.setMaximumSize(QtCore.QSize(200, 247))
        self.label.setObjectName("label")

        self.horizontalLayout_4.addWidget(self.label)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")

        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_4.setMinimumSize(QtCore.QSize(200, 0))
        self.label_4.setMaximumSize(QtCore.QSize(200, 247))
        self.label_4.setObjectName("label_4")

        self.horizontalLayout_3.addWidget(self.label_4)

        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button_gesture_detection.setText(_translate("MainWindow", "手势检测"))
        self.button_gesture_recognition.setText(_translate("MainWindow", "手势识别"))
        self.button_gesture_score.setText(_translate("MainWindow", "手势评分"))
        self.label_5.setText(_translate("MainWindow", ""))
        self.label_6.setText(_translate("MainWindow", ""))
        self.label.setText(_translate("MainWindow", ""))
        self.label_4.setText(_translate("MainWindow", ""))

        self.setWindowTitle("肖帮")  # 设置标题
        self.setWindowIcon(QIcon('img/piano.png'))  # 设置logo

        self.button_gesture_recognition.setStyleSheet('QPushButton {background-color: #c29194}')
        self.button_gesture_detection.setStyleSheet('QPushButton {background-color: #c29194}')
        self.button_gesture_score.setStyleSheet('QPushButton {background-color: #c29194}')

    def paintEvent(self, event):  # 设置背景图
        painter = QPainter(self)
        pixmap = QPixmap("img/chopin.png")
        # 绘制窗口背景，平铺到整个窗口，随着窗口改变而改变
        painter.drawPixmap(self.rect(), pixmap)
