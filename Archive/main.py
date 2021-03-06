# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(650, 750)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setMinimumSize(QtCore.QSize(650, 750))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        Dialog.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Waveguide.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Dialog.setWindowIcon(icon)
        Dialog.setToolTip("")
        Dialog.setModal(False)
        self.gridLayout_4 = QtWidgets.QGridLayout(Dialog)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label = QtWidgets.QLabel(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 0, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(0, 703, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem, 0, 1, 4, 1)
        self.widget_2 = QtWidgets.QWidget(Dialog)
        self.widget_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.widget_2.setObjectName("widget_2")
        self.gridLayout_3.addWidget(self.widget_2, 1, 0, 1, 1)
        self.widget = QtWidgets.QWidget(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.widget.setObjectName("widget")
        self.gridLayout_3.addWidget(self.widget, 2, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.groupBox.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.pushButton_SaveTxt = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_SaveTxt.setMinimumSize(QtCore.QSize(0, 50))
        self.pushButton_SaveTxt.setObjectName("pushButton_SaveTxt")
        self.gridLayout_2.addWidget(self.pushButton_SaveTxt, 1, 1, 1, 1)
        self.pushButton_GenerateWaveguide = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_GenerateWaveguide.setMinimumSize(QtCore.QSize(0, 50))
        self.pushButton_GenerateWaveguide.setObjectName("pushButton_GenerateWaveguide")
        self.gridLayout_2.addWidget(self.pushButton_GenerateWaveguide, 1, 0, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.lineEdit_throat_diameter = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_throat_diameter.setObjectName("lineEdit_throat_diameter")
        self.gridLayout.addWidget(self.lineEdit_throat_diameter, 0, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 0, 2, 1, 1)
        self.lineEdit_width = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_width.setObjectName("lineEdit_width")
        self.gridLayout.addWidget(self.lineEdit_width, 0, 3, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.lineEdit_angle_factor = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_angle_factor.setObjectName("lineEdit_angle_factor")
        self.gridLayout.addWidget(self.lineEdit_angle_factor, 1, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 1, 2, 1, 1)
        self.lineEdit_height = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_height.setObjectName("lineEdit_height")
        self.gridLayout.addWidget(self.lineEdit_height, 1, 3, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 2, 0, 1, 1)
        self.lineEdit_coverage_angle = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_coverage_angle.setReadOnly(True)
        self.lineEdit_coverage_angle.setObjectName("lineEdit_coverage_angle")
        self.gridLayout.addWidget(self.lineEdit_coverage_angle, 2, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 2, 2, 1, 1)
        self.lineEdit_depth_factor = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_depth_factor.setObjectName("lineEdit_depth_factor")
        self.gridLayout.addWidget(self.lineEdit_depth_factor, 2, 3, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 2)
        self.gridLayout_5.addLayout(self.gridLayout_2, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.groupBox, 3, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem1, 4, 0, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Waveguide Designer"))
        self.label.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Waveguide Designer</span></p></body></html>"))
        self.groupBox.setTitle(_translate("Dialog", "Parameters"))
        self.pushButton_SaveTxt.setText(_translate("Dialog", "Save Output"))
        self.pushButton_GenerateWaveguide.setText(_translate("Dialog", "Generate Waveguide"))
        self.label_2.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:9pt; font-weight:600;\">Throat Diameter (mm)</span></p></body></html>"))
        self.lineEdit_throat_diameter.setToolTip(_translate("Dialog", "<html><head/><body><p>Enter Throat Diameter in mm</p></body></html>"))
        self.label_5.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:9pt; font-weight:600;\">Width (mm)</span></p></body></html>"))
        self.lineEdit_width.setToolTip(_translate("Dialog", "<html><head/><body><p>Enter Waveguide Width in mm</p></body></html>"))
        self.label_3.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-size:9pt; font-weight:600;\">Angle Factor</span></p></body></html>"))
        self.lineEdit_angle_factor.setToolTip(_translate("Dialog", "<html><head/><body><p>Adjust the Angle Factor until the desired coverage angle is reached</p></body></html>"))
        self.label_6.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:9pt; font-weight:600;\">Height (mm)</span></p></body></html>"))
        self.lineEdit_height.setToolTip(_translate("Dialog", "<html><head/><body><p>Enter Waveguide Height in mm, if the waveguide is circular use the same number as Width</p></body></html>"))
        self.label_4.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-size:9pt; font-weight:600;\">Coverage Angle (deg)</span></p></body></html>"))
        self.lineEdit_coverage_angle.setToolTip(_translate("Dialog", "<html><head/><body><p>Coverage Angle will appear here after each calculation</p></body></html>"))
        self.label_7.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:9pt; font-weight:600;\">Depth Factor</span></p></body></html>"))
        self.lineEdit_depth_factor.setToolTip(_translate("Dialog", "<html><head/><body><p>Adjust the Depth Factor until desired height is reached.</p><p>Note: Must be above 2</p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
