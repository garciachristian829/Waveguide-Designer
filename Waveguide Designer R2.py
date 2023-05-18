import distutils.util
import sys

# import PyQt5.QtWidgets
import numpy as np
import pyvista
import pyvista as pv
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import *

from pyvistaqt import QtInteractor
from qtpy import QtWidgets

# from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from pyvistaGUI import Ui_MainWindow
from cross_section_ui import Ui_Dialog


class CrossSectionWindow(QDialog, Ui_Dialog):

    def __init__(self, ver_cross_array, hor_cross_array, wg_height, phaseplug_array=None):
        super().__init__()
        if phaseplug_array is None:
            phaseplug_array = []

        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('Waveguide_Designer.ico'))
        self.setWindowTitle("Waveguide Cross Section")
        self.horGraph = pg.PlotWidget()
        self.verGraph = pg.PlotWidget()

        self.horGraph.clear()
        self.verGraph.clear()

        styles_label = {'color': 'k', 'font-size': '13px'}

        self.horGraph.setTitle("Horizontal Cross-Section", size="25pt")
        self.horGraph.setLabel('bottom', 'Width (mm)', **styles_label)
        self.verGraph.setTitle("Vertical Cross-Section", size="25pt", )
        self.verGraph.setLabel('bottom', 'Width (mm)', **styles_label)

        self.gridLayout_2.addWidget(self.horGraph)
        self.gridLayout_2.addWidget(self.verGraph)

        pen = pg.mkPen(color=(255, 0, 0), width=6, style=QtCore.Qt.SolidLine)

        self.horGraph.setBackground('w')
        self.horGraph.showGrid(x=True, y=True)
        self.horGraph.hideButtons()
        self.horGraph.setMouseEnabled(x=False, y=False)
        self.horGraph.setMenuEnabled(False)

        self.verGraph.setBackground('w')
        self.verGraph.showGrid(x=True, y=True)
        self.verGraph.hideButtons()
        self.verGraph.setMouseEnabled(x=False, y=False)
        self.verGraph.setMenuEnabled(False)

        xmax_hor = np.max(hor_cross_array, axis=0)
        ymax_ver = np.max(ver_cross_array, axis=0)

        # Used to check which x value is larger and match the X axis for both plots
        if xmax_hor[0] >= ymax_ver[1]:
            self.horGraph.setXRange(xmax_hor[0] * -1, xmax_hor[0], padding=0)
            self.verGraph.setXRange(xmax_hor[0] * -1, xmax_hor[0], padding=0)
        else:
            self.horGraph.setXRange(ymax_ver[1] * -1, ymax_ver[1], padding=0)
            self.verGraph.setXRange(ymax_ver[1] * -1, ymax_ver[1], padding=0)

        self.verGraph.plot(ver_cross_array[:, 1], ver_cross_array[:, 2], pen=pen)
        self.verGraph.plot((ver_cross_array[:, 1] * -1), ver_cross_array[:, 2], pen=pen)
        self.horGraph.plot(hor_cross_array[:, 0], hor_cross_array[:, 2], pen=pen)
        self.horGraph.plot((hor_cross_array[:, 0] * -1), hor_cross_array[:, 2], pen=pen)

        height_text = pg.TextItem(text='Height: {} mm'.format(wg_height), anchor=(2, 1), color=(0, 0, 0))
        self.verGraph.addItem(height_text)
        if len(phaseplug_array) == 0:
            pass

        else:
            self.verGraph.plot(phaseplug_array[:, 0], phaseplug_array[:, 2], pen=pen)
            self.verGraph.plot(phaseplug_array[:, 0] * -1, phaseplug_array[:, 2], pen=pen)
            self.horGraph.plot(phaseplug_array[:, 0], phaseplug_array[:, 2], pen=pen)
            self.horGraph.plot(phaseplug_array[:, 0] * -1, phaseplug_array[:, 2], pen=pen)

    def closeEvent(self, QCloseEvent):
        super().closeEvent(QCloseEvent)
        self.close()


class MyMainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        # Define plotter and then attach to gridlayout to allow resizing with window, add axes to identify xyz
        self.plotter = QtInteractor(self.frame)
        self.plotter.set_background(color='white')
        self.plotter.show_axes()
        self.gridLayout_5.addWidget(self.plotter.interactor)

        self.setWindowIcon(QtGui.QIcon('Waveguide_Designer.ico'))

        # Set checkbox to not checked
        self.groupBox_phaseplug.setEnabled(False)
        # Define buttons and checkbox state check
        self.pushButton_generate_waveguide.clicked.connect(self.generate_waveguide)
        self.pushButton_save_button.clicked.connect(self.on_click2)
        self.checkBox_phaseplug.stateChanged.connect(self.check_state)
        self.actionSave_Waveguide_Parameters.triggered.connect(self.parameters_save)
        self.actionLoad_Waveguide_Parameters.triggered.connect(self.parameters_load)
        self.actionSave_Comsol_Parameters.triggered.connect(self.comsol_parameters)
        self.actionExport_OBJ.triggered.connect(self.export_obj)

        # Set type of number that can be accepted in all input boxes (Double)
        self.lineEdit_throat_diameter.setValidator(
            QtGui.QDoubleValidator(notation=QtGui.QDoubleValidator.StandardNotation))
        self.lineEdit_width.setValidator(QtGui.QDoubleValidator(notation=QtGui.QDoubleValidator.StandardNotation))
        self.lineEdit_height.setValidator(QtGui.QDoubleValidator(notation=QtGui.QDoubleValidator.StandardNotation))
        self.lineEdit_depth_factor.setValidator(
            QtGui.QDoubleValidator(notation=QtGui.QDoubleValidator.StandardNotation))
        self.lineEdit_plugoffset.setValidator(QtGui.QDoubleValidator(notation=QtGui.QDoubleValidator.StandardNotation))
        self.lineEdit_dome_diameter.setValidator(
            QtGui.QDoubleValidator(notation=QtGui.QDoubleValidator.StandardNotation))
        self.lineEdit_plug_diameter.setValidator(
            QtGui.QDoubleValidator(notation=QtGui.QDoubleValidator.StandardNotation))
        self.lineEdit_angle_factor.setValidator(
            QtGui.QDoubleValidator(notation=QtGui.QDoubleValidator.StandardNotation))

        self.show()

    def generate_waveguide(self):
        # Clear plotter each time to plot new waveguides
        self.plotter.clear()

        # Get Parameters from LineEdits in Waveguide Groupbox
        throat_diameter = float(self.lineEdit_throat_diameter.text())  # (1)
        angle_factor = float(self.lineEdit_angle_factor.text())  # (2)
        width = float(self.lineEdit_width.text())  # (3)
        height = float(self.lineEdit_height.text())  # (4)
        depth_factor = float(self.lineEdit_depth_factor.text())  # (5)

        if self.radioButton_elliptical.isChecked():
            self.circle_array, self.ellipse_array, self.center_array, self.ver_array, self.hor_array, \
            waveguide_mesh \
                = self.elliptical_calc(throat_diameter, width, height, depth_factor, angle_factor)

        elif self.radioButton_rectangular.isChecked():
            self.circle_array, self.x_rectangle, self.y_rectangle, self.center_array, self.ver_array, self.hor_array, \
            waveguide_mesh, self.x_midline, self.y_midline \
                = self.rectangular_calc(throat_diameter, width, height, depth_factor, angle_factor)

        # Calculate zmax of waveguide
        z_max = round(np.max(self.ver_array[:, 2]), 2)

        # calculate coverage angles using averages of 1/3 -> 2/3 method
        first_section = round(z_max / 3)
        second_section = first_section * 2
        nearthroat_value = self.find_nearest(self.ver_array[:, 2], first_section)
        nearmouth_value = self.find_nearest(self.ver_array[:, 2], second_section)
        nearthroat_index = np.where(self.ver_array == nearthroat_value)[0][0]
        nearmouth_index = np.where(self.ver_array == nearmouth_value)[0][0]

        ver_coverage_angle_list = []
        hor_coverage_angle_list = []
        for i in range(nearthroat_index, nearmouth_index + 1):
            ver_coverage_angle_list.append(self.coverage_calc(self.ver_array[i, 1], self.ver_array[i, 2],
                                                              self.ver_array[i + 2, 1], self.ver_array[i + 2, 2]))

            hor_coverage_angle_list.append(self.coverage_calc(self.hor_array[i, 0], self.hor_array[i, 2],
                                                              self.hor_array[i + 2, 0], self.hor_array[i + 2, 2]))

        hor_calc_coverage_angle = sum(hor_coverage_angle_list) / len(hor_coverage_angle_list)
        ver_calc_coverage_angle = sum(ver_coverage_angle_list) / len(ver_coverage_angle_list)

        # Set all values in array less than 1 to 0
        self.hor_array[self.hor_array < 1] = 0
        self.ver_array[self.ver_array < 1] = 0

        # Calculate cutoff freq during function calc
        cutoff_freq = self.cutoff_frequency(hor_calc_coverage_angle, throat_diameter)

        # Variables that will be used to output on results tab
        self.lineEdit_coverage_angle.setText((str(int(hor_calc_coverage_angle)) + ' deg'))
        self.lineEdit_ver_coverage_angle.setText((str(int(ver_calc_coverage_angle)) + ' deg'))
        self.lineEdit_cutoff_freq.setText((str(int(cutoff_freq)) + ' Hz'))

        # Reflect mesh twice and merge twice to create a entire surface
        waveguide_mesh_reflected = waveguide_mesh.reflect((0, 1, 0))
        merged = waveguide_mesh.merge(waveguide_mesh_reflected)
        merged_mirror = merged.reflect((1, 0, 0))
        waveguide_mesh = merged.merge(merged_mirror)

        # Select scalars to plot "cmap" colors without needing matplotlib
        waveguide_mesh['Data'] = waveguide_mesh.points[:, 2]

        self.waveguide_whole = waveguide_mesh

        if not (self.checkBox_phaseplug.isChecked()):
            # If phaseplug checkbox is not checked, pass an empty array to avoid any issues
            self.phaseplug_array = np.array([])

            self.plotter.add_mesh(waveguide_mesh, show_scalar_bar=False)

            self.dlg = CrossSectionWindow(self.ver_array, self.hor_array, z_max)
            self.dlg.show()

        else:
            # If checkbox is checked, gather data from PhasePlug Groupbox
            phase_plug_dia = float(self.lineEdit_plug_diameter.text())
            dome_dia = float(self.lineEdit_dome_diameter.text())
            phase_plug_offset = float(self.lineEdit_plugoffset.text())

            if dome_dia < phase_plug_dia:
                self.error_phaseplug()
                return

            else:
                # Calculate phaseplug mesh and array
                phaseplug_mesh, self.phaseplug_array = self.phase_plug_calc(phase_plug_dia, dome_dia, phase_plug_offset,
                                                                            array_length=100)
                # mirror meshes and form phasing plug
                phaseplug_mesh_reflected = phaseplug_mesh.reflect((0, 1, 0))
                merged_mesh = phaseplug_mesh.merge(phaseplug_mesh_reflected)
                merged_mirror = merged_mesh.reflect((1, 0, 0))
                phaseplug_mesh = merged_mesh.merge(merged_mirror)

                phaseplug_mesh['Data'] = phaseplug_mesh.points[:, 2]

                self.plotter.add_mesh(waveguide_mesh, show_scalar_bar=False)
                self.plotter.add_mesh(phaseplug_mesh, show_scalar_bar=False)

                self.dlg = CrossSectionWindow(self.ver_array, self.hor_array, z_max, self.phaseplug_array)
                self.dlg.show()

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def on_click2(self):
        # Button to save text to folder. Folder is specified by user.

        save_text = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))

        if save_text == '':
            self.no_file_selected()
        else:

            np.savetxt(save_text + "/Throat.txt", self.circle_array, delimiter=" ")
            np.savetxt(save_text + "/Horizontal.txt", self.hor_array, delimiter=" ")
            np.savetxt(save_text + "/Vertical.txt", self.ver_array, delimiter=" ")
            np.savetxt(save_text + "/Center.txt", self.center_array, delimiter=" ")

            if self.radioButton_elliptical.isChecked():
                np.savetxt(save_text + "/Ellipse.txt", self.ellipse_array, delimiter=" ")
                if self.checkBox_phaseplug.isChecked():
                    np.savetxt(save_text + "/Phaseplug.txt", self.phaseplug_array, delimiter=" ")

            elif self.radioButton_rectangular.isChecked():
                np.savetxt(save_text + "/X_Rectangle.txt", self.x_rectangle, delimiter=" ")
                np.savetxt(save_text + "/Y_Rectangle.txt", self.y_rectangle, delimiter=" ")
                np.savetxt(save_text + "/X_mid_Rectangle.txt", self.x_midline, delimiter=" ")
                np.savetxt(save_text + "/Y_mid_Rectangle.txt", self.y_midline, delimiter=" ")

                if self.checkBox_phaseplug.isChecked():
                    np.savetxt(save_text + "/Phaseplug.txt", self.phaseplug_array, delimiter=" ")

    def check_state(self, state):
        # This function will check the state of the phaseplug_checkbox and if it is checked it will enable it and accept
        # input, otherwise it greys out the inputs and clears the content
        if state == 0:
            self.groupBox_phaseplug.setEnabled(False)
            self.lineEdit_dome_diameter.clear()
            self.lineEdit_plug_diameter.clear()
            self.lineEdit_plugoffset.clear()
        else:
            self.groupBox_phaseplug.setEnabled(True)

    def parameters_save(self):
        file_name = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", '/', "WGP (*.wgp)")[0]

        if file_name == '':
            self.no_file_selected()

        else:

            parameter_list = [str(self.lineEdit_throat_diameter.text()) + "\n",
                              str(self.lineEdit_width.text()) + "\n",
                              str(self.lineEdit_height.text()) + "\n",
                              str(self.lineEdit_angle_factor.text()) + "\n",
                              str(self.lineEdit_depth_factor.text()) + "\n",
                              str(self.checkBox_phaseplug.isChecked()) + "\n",
                              str(self.radioButton_elliptical.isChecked()) + "\n",
                              str(self.radioButton_rectangular.isChecked()) + "\n"
                              ]

            if self.checkBox_phaseplug.isChecked():
                parameter_phaseplug = [
                    str(self.lineEdit_plug_diameter.text()) + "\n",
                    str(self.lineEdit_dome_diameter.text()) + "\n",
                    str(self.lineEdit_plugoffset.text())
                ]
                parameter_list.extend(parameter_phaseplug)

            file_parameters = open(file_name, "w")
            file_parameters.writelines(parameter_list)

    def parameters_load(self):
        parameter_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Waveguide Parameter File",
                                                               "", 'WGP (*.wgp)')[0]
        if parameter_file == '':
            self.no_file_selected()

        else:

            parameters = open(parameter_file, "r")

            content = parameters.read().splitlines()

            self.lineEdit_throat_diameter.setText(content[0])
            self.lineEdit_width.setText(content[1])
            self.lineEdit_height.setText(content[2])
            self.lineEdit_angle_factor.setText(content[3])
            self.lineEdit_depth_factor.setText(content[4])
            self.checkBox_phaseplug.setChecked(bool(distutils.util.strtobool(content[5])))
            self.radioButton_elliptical.setChecked(bool(distutils.util.strtobool(content[6])))
            self.radioButton_rectangular.setChecked(bool(distutils.util.strtobool(content[7])))

            if bool(distutils.util.strtobool(content[5])):
                self.lineEdit_plug_diameter.setText(content[8])
                self.lineEdit_dome_diameter.setText(content[9])
                self.lineEdit_plugoffset.setText(content[10])

            self.pushButton_generate_waveguide.click()

    def comsol_parameters(self):
        file_name = QtWidgets.QFileDialog.getSaveFileName(self, "Save Comsol Parameters", '/', "txt (*.txt)")[0]

        if file_name == '':
            self.no_file_selected()

        else:
            comsol_params = [
                "x_ellipse " + str(self.lineEdit_width.text()) + "[mm] Ellipse X " + "\n",
                "y_ellipse " + str(self.lineEdit_height.text()) + "[mm] Ellipse Y " + "\n",
                "throat " + str(self.lineEdit_throat_diameter.text()) + "[mm] Waveguide Throat " + "\n",
                "depth_factor " + str(self.lineEdit_depth_factor.text()) + " depth factor " + "\n",
                "angle_factor " + str(self.lineEdit_angle_factor.text()) + " angle factor " + "\n"
            ]
            if self.checkBox_phaseplug.isChecked():
                comsol_phaseplug = [
                    "plug_dia " + str(self.lineEdit_plug_diameter.text()) + "[mm] phase plug diameter " + "\n",
                    "dome_dia " + str(self.lineEdit_dome_diameter.text()) + "[mm] tweeter dome diameter " + "\n",
                    "plug_offset " + str(self.lineEdit_plugoffset.text()) + "[mm] phase plug offset " + "\n"
                ]
                comsol_params.extend(comsol_phaseplug)
            else:
                comsol_phaseplug = [
                    "plug_dia 0 [mm] phase plug diameter " + "\n",
                    "dome_dia 0 [mm] tweeter dome diameter " + "\n",
                    "plug_offset 0 [mm] phase plug offset " + "\n"
                ]
                comsol_params.extend(comsol_phaseplug)

            file_parameters = open(file_name, "w")
            file_parameters.writelines(comsol_params)

    def coverage_calc(self, x_1, y_1, x_2, y_2):
        # Calculate Slope
        slope = (y_1 - y_2) / (x_1 - x_2)

        angle = np.degrees(np.arctan(slope))

        coverage_angle = (180 - (angle * 2)) * 0.5

        return coverage_angle

    def elliptical_calc(self, waveguide_throat, ellipse_x, ellipse_y, depth_factor, angle_factor):
        array_length = 100
        # now create the actual structured grid
        # 2d circular grid
        r, phi = np.mgrid[0:1:array_length * 1j, 0:np.pi / 2:array_length * 1j]

        # transform to ellipse on the outside, circle on the inside
        x = (ellipse_x / 2 * r + waveguide_throat / 2 * (1 - r)) * np.cos(phi)
        y = (ellipse_y / 2 * r + waveguide_throat / 2 * (1 - r)) * np.sin(phi)

        # compute z profile
        angle_factor = angle_factor / 10000
        z = (ellipse_x / 2 * r / angle_factor) ** (1 / depth_factor)

        throat = np.array(np.column_stack((x[0, 0:array_length], y[0, 0:array_length],
                                           z[0, 0:array_length])))
        ellipse = np.array(np.column_stack((x[array_length - 1, 0:array_length], y[array_length - 1, 0:array_length],
                                            z[array_length - 1, 0:array_length])))
        horizontal_line = np.array(
            np.column_stack((x[0:array_length, 0], y[0:array_length, 0], z[0:array_length, 0])))
        vertical_line = np.array(
            np.column_stack((x[0:array_length, array_length - 1], y[0:array_length, array_length - 1],
                             z[0:array_length, array_length - 1])))
        center_line = np.array(np.column_stack((x[0:array_length, 50], y[0:array_length, 50], z[0:array_length, 50])))

        elliptical_mesh = pv.StructuredGrid(x, y, z)

        return throat, ellipse, center_line, vertical_line, horizontal_line, elliptical_mesh

    def rectangular_calc(self, waveguide_throat, rectangle_x, rectangle_y, depth_factor, angle_factor):
        array_length = 100
        rectangle_x = rectangle_x / 2
        rectangle_y = rectangle_y / 2
        # waveguide throat line
        phi = np.linspace(0, np.pi / 2, array_length)
        x_interior = waveguide_throat / 2 * np.cos(phi)
        y_interior = waveguide_throat / 2 * np.sin(phi)

        # theta is angle where x and y intersect
        theta = np.arctan2(rectangle_y, rectangle_x)
        # find array index which maps to corner of rectangle
        corner_index = (array_length * (2 * theta / np.pi)).round().astype(int)
        # construct rectangular coordinate manually
        x_exterior = np.zeros_like(x_interior)
        y_exterior = x_exterior.copy()
        phi_aux = np.linspace(0, theta, corner_index)
        x_exterior[:corner_index] = rectangle_x
        y_exterior[:corner_index] = rectangle_x * np.tan(phi_aux)
        phi_aux = np.linspace(np.pi / 2, theta, array_length - corner_index, endpoint=False)[::-1]
        x_exterior[corner_index:] = rectangle_y / np.tan(phi_aux)
        y_exterior[corner_index:] = rectangle_y

        # interpolate between two curves
        r = np.linspace(0, 1, array_length)[:, None]  # shape (array_length, 1) for broadcasting
        x = x_exterior * r + x_interior * (1 - r)
        y = y_exterior * r + y_interior * (1 - r)

        # compute z profile
        angle_factor = angle_factor / 10000
        z = (rectangle_x / 2 * r / angle_factor) ** (1 / depth_factor)
        # explicitly broadcast to the shape of x and y
        z = np.broadcast_to(z, x.shape)

        # prepare data for export
        throat = np.array(np.column_stack((x[0, 0:array_length], y[0, 0:array_length],
                                           z[0, 0:array_length])))
        x_rectangle = np.array(np.column_stack((x[array_length - 1, 0:corner_index],
                                                y[array_length - 1, 0:corner_index],
                                                z[array_length - 1, 0:corner_index])))

        y_rectangle = np.array(np.column_stack((x[array_length - 1, corner_index - 1:array_length],
                                                y[array_length - 1, corner_index - 1:array_length],
                                                z[array_length - 1, corner_index - 1:array_length])))

        horizontal_line = np.array(
            np.column_stack((x[0:array_length, 0], y[0:array_length, 0], z[0:array_length, 0])))

        vertical_line = np.array(
            np.column_stack((x[0:array_length, array_length - 1], y[0:array_length, array_length - 1],
                             z[0:array_length, array_length - 1])))

        center_line = np.array(
            np.column_stack((x[0:array_length, corner_index - 1], y[0:array_length, corner_index - 1],
                             z[0:array_length, corner_index - 1])))

        y_mid_center_line = np.array(np.column_stack((x[0: array_length, int(corner_index / 2)],
                                                      y[0: array_length, int(corner_index / 2)],
                                                      z[0: array_length, int(corner_index / 2)])))

        x_mid_center_line = np.array(np.column_stack((
            x[0: array_length, int(array_length - corner_index)],
            y[0: array_length, int(array_length - corner_index)],
            z[0: array_length, int(array_length - corner_index)])))

        # Create mesh for 3D viewing
        rectangular_mesh = pv.StructuredGrid(x, y, z)

        return throat, x_rectangle, y_rectangle, center_line, vertical_line, horizontal_line, rectangular_mesh, \
               x_mid_center_line, y_mid_center_line

    def cutoff_frequency(self, coverage_angle, throat_diameter):
        coverage_angle = 0.5 * (coverage_angle * (np.pi / 180))

        throat_radius = (throat_diameter / 2)

        cutoff_freq = (44 * (np.sin(coverage_angle) / (throat_radius / 1000)))

        return cutoff_freq

    def phase_plug_calc(self, plug_dia, dome_dia, plug_offset, array_length):
        s, chi = np.mgrid[0:1:array_length * 1j, 0:np.pi / 2:array_length * 1j]

        # Create phase plug dome dependant on 2 different situations
        if plug_dia == dome_dia:

            x_phaseplug = ((dome_dia / 2 * (1 - s)) * np.cos(chi))
            y_phaseplug = ((dome_dia / 2 * (1 - s)) * np.sin(chi))
            z_phaseplug = np.sqrt(abs((dome_dia / 2) ** 2 - (x_phaseplug ** 2) - (y_phaseplug ** 2))) + plug_offset

        elif plug_dia < dome_dia > 0:

            alpha_angle = 2 * np.arcsin(plug_dia / dome_dia)
            plug_modification = (plug_dia / 2) / (np.tan(alpha_angle / 2))

            x_phaseplug = ((plug_dia / 2 * (1 - s)) * np.cos(chi))
            y_phaseplug = ((plug_dia / 2 * (1 - s)) * np.sin(chi))
            z_phaseplug = np.sqrt(
                abs((dome_dia / 2) ** 2 - (x_phaseplug ** 2) - (y_phaseplug ** 2))) + plug_offset - plug_modification

        phaseplug = pv.StructuredGrid(x_phaseplug, y_phaseplug, z_phaseplug)
        phaseplug_line = np.array(np.column_stack((x_phaseplug[0:array_length, 0], y_phaseplug[0:array_length, 0],
                                                   z_phaseplug[0:array_length, 0])))

        return phaseplug, phaseplug_line

    def no_file_selected(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowIcon(QtGui.QIcon("Waveguide_Designer.ico"))
        msg.setText("No File Was Selected!")
        msg.setWindowTitle("Warning")
        retval = msg.exec_()

    def error_phaseplug(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowIcon(QtGui.QIcon("Waveguide_Designer.ico"))
        msg.setText("Plug diameter cannot be larger then dome diameter!")
        msg.setWindowTitle("Error")
        msg.exec_()

    def export_obj(self):

        stl_filename = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", '/', "STL (*.stl)")[0]

        if stl_filename == '':
            self.no_file_selected()

        else:
            mesh = self.waveguide_whole.triangulate(inplace=True)
            pyvista.save_meshio(stl_filename, mesh)

    def closeEvent(self, QCloseEvent):
        super().closeEvent(QCloseEvent)
        self.plotter.close()
        try:
            self.dlg.close()
        except AttributeError as a:
            pass


if __name__ == "__main__":
    # MAIN APP
    app = QtWidgets.QApplication(sys.argv)
    win = MyMainWindow()
    sys.exit(app.exec_())
