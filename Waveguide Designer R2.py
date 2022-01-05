import numpy as np
import pyvista as pv
import sys

from PyQt5 import QtCore, QtWidgets
from qtpy import QtWidgets
from pyvistaqt import QtInteractor
from pyvistaGUI import (Ui_MainWindow)
# from <filename> of the UI python initialization (content not changed)
from PyQt5.QtCore import pyqtSlot


def coverage_calc(x_1, y_1, x_2, y_2):
    slope = (y_1 - y_2) / (x_1 - x_2)

    angle = np.degrees(np.arctan(slope))

    coverage_angle = 180 - (angle * 2)

    return coverage_angle


def main_calc(waveguide_throat, ellipse_x, ellipse_y, depth_factor, angle_factor):
    array_length = 100

    # now create the actual structured grid
    # 2d circular grid, one for waveguide and the other for phase plug
    r, phi = np.mgrid[0:1:array_length * 1j, 0:np.pi / 2:array_length * 1j]

    # transform to ellipse on the outside, circle on the inside
    x = (ellipse_x / 2 * r + waveguide_throat / 2 * (1 - r)) * np.cos(phi)
    y = (ellipse_y / 2 * r + waveguide_throat / 2 * (1 - r)) * np.sin(phi)

    # compute z profile
    angle_factor = angle_factor / 10000
    z = (ellipse_x / 2 * r / angle_factor) ** (1 / depth_factor)

    waveguide = pv.StructuredGrid(x, y, z)

    throat = np.array([np.column_stack((x[0, 0:array_length], y[0, 0:array_length], z[0, 0:array_length]))])
    ellipse = np.array([np.column_stack((x[array_length - 1, 0:array_length], y[array_length - 1, 0:array_length],
                                         z[array_length - 1, 0:array_length]))])
    horizontal_line = np.array(
        [np.column_stack((x[0:array_length, 0], y[0:array_length, 0], z[0:array_length, 0]))])
    vertical_line = np.array([np.column_stack((x[0:array_length, 0], y[0:array_length, 0], z[0:array_length, 0]))])
    center_line = np.array([np.column_stack((x[0:array_length, 50], y[0:array_length, 50], z[0:array_length, 50]))])

    calc_coverage_angle = coverage_calc(horizontal_line[0, 50, 0], horizontal_line[0, 50, 1],
                                        horizontal_line[0, 51, 0], horizontal_line[0, 51, 1])

    return throat, ellipse, horizontal_line, vertical_line, center_line, calc_coverage_angle, waveguide


def save_text_data(circle_array, ellipse_array, hor_array, ver_array, cen_array, phase_plug, save_text):
    if not phase_plug.any():
        np.savetxt(save_text + "/Throat.txt", circle_array, delimiter=" ")
        np.savetxt(save_text + "/ellipse.txt", ellipse_array, delimiter=" ")
        np.savetxt(save_text + "/hor.txt", hor_array, delimiter=" ")
        np.savetxt(save_text + "/ver.txt", ver_array, delimiter=" ")
        np.savetxt(save_text + "/cen.txt", cen_array, delimiter=" ")

    else:
        np.savetxt(save_text + "/Throat.txt", circle_array, delimiter=" ")
        np.savetxt(save_text + "/ellipse.txt", ellipse_array, delimiter=" ")
        np.savetxt(save_text + "/hor.txt", hor_array, delimiter=" ")
        np.savetxt(save_text + "/ver.txt", ver_array, delimiter=" ")
        np.savetxt(save_text + "/cen.txt", cen_array, delimiter=" ")
        np.savetxt(save_text + "/phase_plug.txt", phase_plug, delimiter=" ")

    return ()


def cutoff_frequency(coverage_angle, throat_diameter):
    coverage_angle = coverage_angle / 2

    throat_radius = (throat_diameter / 2) / 1000

    cutoff_freq = abs((44 * (np.radians(np.sin(coverage_angle)) / throat_radius)) * (-1))

    return cutoff_freq


def phase_plug_calc(plug_dia, dome_dia, plug_offset, array_length):
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
    phaseplug_line = np.array([np.column_stack((x_phaseplug[0:array_length, 50], y_phaseplug[0:array_length, 50],
                                                z_phaseplug[0:array_length, 50]))])

    return phaseplug, phaseplug_line


class MyMainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        # Define plotter and then attach to gridlayout to allow resizing with window
        self.plotter = QtInteractor(self.frame)
        self.gridLayout_5.addWidget(self.plotter.interactor)
        # Set checkbox to not checked
        self.groupBox_phaseplug.setEnabled(False)
        # Define buttons and checkbox state check
        self.pushButton_generate_waveguide.clicked.connect(self.generate_waveguide)
        self.pushButton_save_button.clicked.connect(self.on_click2)
        self.checkBox_phaseplug.stateChanged.connect(self.check_state, self.checkBox_phaseplug.isChecked())

        self.show()

    @pyqtSlot()
    def generate_waveguide(self):
        # Clear plotter each time to plot new waveguides
        self.plotter.clear()
        # Get Parameters from LineEdits
        throat_diameter = float(self.lineEdit_throat_diameter.text())  # (1)
        angle_factor = float(self.lineEdit_angle_factor.text())  # (3)
        width = float(self.lineEdit_width.text())  # (4)
        height = float(self.lineEdit_height.text())  # (5)
        depth_factor = float(self.lineEdit_depth_factor.text())  # (6)

        self.circle_array, self.ellipse_array, self.hor_array, self.ver_array, self.center_array, self.coverage_angle,\
        waveguide_mesh = main_calc(throat_diameter, width, height, depth_factor, angle_factor)

        cutoff_freq = cutoff_frequency(self.coverage_angle, throat_diameter)

        coverage_angle = str(int(self.coverage_angle))
        cutoff_freq = str(int(cutoff_freq))
        self.lineEdit_coverage_angle.setText(coverage_angle)
        self.lineEdit_cutoff_freq.setText(cutoff_freq)

        # Reflect mesh twice and merge twice to create a entire surface
        waveguide_mesh_reflected = waveguide_mesh.reflect((0, 1, 0))
        merged = waveguide_mesh.merge(waveguide_mesh_reflected)
        merged_mirror = merged.reflect((1, 0, 0))
        waveguide_mesh = merged.merge(merged_mirror)
        # Select scalars to plot "cmap" colors without needing matplotlib
        waveguide_mesh['Data'] = waveguide_mesh.points[:, 2]

        if not (self.checkBox_phaseplug.isChecked()):

            self.plotter.add_mesh(waveguide_mesh, show_scalar_bar=False)

            self.show()

        else:

            phase_plug_dia = float(self.lineEdit_plug_diameter.text())
            dome_dia = float(self.lineEdit_dome_diameter.text())
            phase_plug_offset = float(self.lineEdit_plugoffset.text())

            phaseplug_mesh, self.phaseplug_array = phase_plug_calc(phase_plug_dia, dome_dia, phase_plug_offset,
                                                                   array_length=100)

            phaseplug_mesh_reflected = phaseplug_mesh.reflect((0, 1, 0))
            merged = phaseplug_mesh.merge(phaseplug_mesh_reflected)
            merged_mirror = merged.reflect((1, 0, 0))
            phaseplug_mesh = merged.merge(merged_mirror)

            phaseplug_mesh['Data'] = phaseplug_mesh.points[:, 2]

            self.plotter.add_mesh(waveguide_mesh, show_scalar_bar=False)
            self.plotter.add_mesh(phaseplug_mesh, show_scalar_bar=False)

            self.show()

    def on_click2(self):
        circle_array = self.circle_array
        ellipse_array = self.ellipse_array
        hor_array = self.hor_array
        ver_array = self.ver_array
        cen_array = self.center_array
        phase_plug = self.phaseplug_array

        save_text = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))

        save_text_data(circle_array, ellipse_array, hor_array, ver_array, cen_array, phase_plug, save_text)

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

        return state


if __name__ == "__main__":
    # MAIN APP
    app = QtWidgets.QApplication(sys.argv)
    win = MyMainWindow()
    sys.exit(app.exec_())
