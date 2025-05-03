import sys
import numpy as np
import pyvista as pv
import pyqtgraph as pg
import logging
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import *
from pyvistaqt import QtInteractor
from pyvistaGUI import Ui_MainWindow
from cross_section_ui import Ui_Dialog

# Constants
ARRAY_LENGTH = 100
ANGLE_FACTOR_SCALE = 10000
COVERAGE_SECTION_RATIO = 1 / 3
LABEL_OFFSET_PIXELS = 50

logging.basicConfig(level=logging.INFO)


def str_to_bool(s):
    return s.lower() in ['true', '1', 'yes']


def mirror_mesh(mesh):
    reflected = mesh.reflect((0, 1, 0))
    merged = mesh.merge(reflected)
    merged_mirror = merged.reflect((1, 0, 0))
    return merged.merge(merged_mirror)


def instantaneous_coverage(x_array, z_array):
    idx = np.argsort(z_array)
    x_array = x_array[idx]
    z_array = z_array[idx]
    dx_dz = np.gradient(x_array, z_array)
    instantaneous_angles = 2 * np.degrees(np.arctan(np.abs(dx_dz)))
    return z_array, instantaneous_angles


def label_theta_curve(graph, z_array, theta_array, label_prefix="θ", num_labels=6, offset_pixels=LABEL_OFFSET_PIXELS):
    idx = np.argsort(z_array)
    z_array = z_array[idx]
    theta_array = theta_array[idx]

    # Calculate arc length along the curve
    dz = np.diff(z_array)
    dtheta = np.diff(theta_array)
    dist = np.sqrt(dz ** 2 + dtheta ** 2)
    arc_length = np.insert(np.cumsum(dist), 0, 0)
    total_length = arc_length[-1]

    # Choose evenly spaced arc lengths
    label_positions = np.linspace(0, total_length, num_labels + 2)[1:-1]
    indices = [np.searchsorted(arc_length, pos) for pos in label_positions]

    vb = graph.getViewBox()
    offset_units = vb.viewPixelSize()[0] * offset_pixels if vb else 5

    for index in indices:
        if index >= len(z_array):
            continue
        z = z_array[index]
        theta = theta_array[index]
        label = pg.TextItem(text=f"{label_prefix}={theta:.1f}°", anchor=(0, 0.5), color=(0, 0, 255))
        label.setPos(z + offset_units, theta)
        graph.addItem(label)

        line = pg.PlotDataItem([z, z + offset_units], [theta, theta],
                               pen=pg.mkPen(color=(0, 0, 255), style=QtCore.Qt.DotLine))
        graph.addItem(line)


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
        self.gridLayout_2.addWidget(self.horGraph)
        self.gridLayout_2.addWidget(self.verGraph)

        styles_label = {'color': 'k', 'font-size': '13px'}
        pen = pg.mkPen(color=(255, 0, 0), width=6, style=QtCore.Qt.SolidLine)

        for graph, title in zip([self.horGraph, self.verGraph], ["Horizontal Cross-Section", "Vertical Cross-Section"]):
            graph.clear()
            graph.setBackground('w')
            graph.showGrid(x=True, y=True)
            graph.hideButtons()
            graph.setMouseEnabled(x=False, y=False)
            graph.setMenuEnabled(False)
            graph.setTitle(title, size="25pt")
            graph.setLabel('bottom', 'Width (mm)', **styles_label)

        max_x_range = max(np.max(hor_cross_array, axis=0)[0], np.max(ver_cross_array, axis=0)[1])
        for graph in [self.horGraph, self.verGraph]:
            graph.setXRange(-max_x_range, max_x_range, padding=0)

        self.verGraph.plot(ver_cross_array[:, 1], ver_cross_array[:, 2], pen=pen)
        self.verGraph.plot(-ver_cross_array[:, 1], ver_cross_array[:, 2], pen=pen)
        self.horGraph.plot(hor_cross_array[:, 0], hor_cross_array[:, 2], pen=pen)
        self.horGraph.plot(-hor_cross_array[:, 0], hor_cross_array[:, 2], pen=pen)

        if len(phaseplug_array):
            self.verGraph.plot(phaseplug_array[:, 0], phaseplug_array[:, 2], pen=pen)
            self.verGraph.plot(-phaseplug_array[:, 0], phaseplug_array[:, 2], pen=pen)
            self.horGraph.plot(phaseplug_array[:, 0], phaseplug_array[:, 2], pen=pen)
            self.horGraph.plot(-phaseplug_array[:, 0], phaseplug_array[:, 2], pen=pen)

        height_text = pg.TextItem(text=f'Height: {wg_height} mm', anchor=(2, 1), color=(0, 0, 0))
        self.verGraph.addItem(height_text)

        # Vertical Instantaneous Coverage Plot
        z_ver, theta_ver = instantaneous_coverage(ver_cross_array[:, 1], ver_cross_array[:, 2])
        pen_angle = pg.mkPen(color=(0, 0, 255), width=2, style=QtCore.Qt.DashLine)
        self.verGraph.plot(z_ver, theta_ver, pen=pen_angle)
        label_theta_curve(self.verGraph, z_ver, theta_ver, label_prefix="θV")

        # Horizontal Instantaneous Coverage Plot
        z_hor, theta_hor = instantaneous_coverage(hor_cross_array[:, 0], hor_cross_array[:, 2])
        self.horGraph.plot(z_hor, theta_hor, pen=pen_angle)
        label_theta_curve(self.horGraph, z_hor, theta_hor, label_prefix="θH")

        # Optional: Label
        angle_label_ver = pg.TextItem(text=f"Instantaneous θ (V)", anchor=(1, 0), color=(0, 0, 255))
        self.verGraph.addItem(angle_label_ver)
        angle_label_hor = pg.TextItem(text=f"Instantaneous θ (H)", anchor=(1, 0), color=(0, 0, 255))
        self.horGraph.addItem(angle_label_hor)

    def closeEvent(self, QCloseEvent):
        super().closeEvent(QCloseEvent)
        self.close()


class MyMainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
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

        # Set validators
        for le in [self.lineEdit_throat_diameter, self.lineEdit_width, self.lineEdit_height,
                   self.lineEdit_depth_factor, self.lineEdit_plugoffset, self.lineEdit_dome_diameter,
                   self.lineEdit_plug_diameter, self.lineEdit_angle_factor]:
            le.setValidator(QtGui.QDoubleValidator(notation=QtGui.QDoubleValidator.StandardNotation))

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

        waveguide_mesh = mirror_mesh(waveguide_mesh)

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
                self.show_warning("Plug Diameter cannot be larger than dom diameter!", "Error")
                return

            else:
                # Calculate phaseplug mesh and array
                phaseplug_mesh, self.phaseplug_array = self.phase_plug_calc(phase_plug_dia, dome_dia, phase_plug_offset,
                                                                            ARRAY_LENGTH=100)
                # mirror meshes and form phasing plug
                phaseplug_mesh = mirror_mesh(phaseplug_mesh)
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
            self.show_warning("No File Was Selected", "Warning")
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
            self.show_warning("No File Was Selected", "Warning")

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

            with open(file_name, "w") as f:
                f.writelines(parameter_list)

    def parameters_load(self):
        parameter_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Waveguide Parameter File",
                                                               "", 'WGP (*.wgp)')[0]
        if parameter_file == '':
            self.show_warning("No File Was Selected", "Warning")

        else:

            parameters = open(parameter_file, "r")

            content = parameters.read().splitlines()

            self.lineEdit_throat_diameter.setText(content[0])
            self.lineEdit_width.setText(content[1])
            self.lineEdit_height.setText(content[2])
            self.lineEdit_angle_factor.setText(content[3])
            self.lineEdit_depth_factor.setText(content[4])
            self.checkBox_phaseplug.setChecked(bool(str_to_bool(content[5])))
            self.radioButton_elliptical.setChecked(bool(str_to_bool(content[6])))
            self.radioButton_rectangular.setChecked(bool(str_to_bool(content[7])))

            if bool(str_to_bool(content[5])):
                self.lineEdit_plug_diameter.setText(content[8])
                self.lineEdit_dome_diameter.setText(content[9])
                self.lineEdit_plugoffset.setText(content[10])

            self.pushButton_generate_waveguide.click()

    def comsol_parameters(self):
        file_name = QtWidgets.QFileDialog.getSaveFileName(self, "Save Comsol Parameters", '/', "txt (*.txt)")[0]

        if file_name == '':
            self.show_warning("No File Was Selected", "Warning")

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

    def elliptical_calc(self, waveguide_throat, ellipse_x, ellipse_y, depth_factor, angle_factor):

        # now create the actual structured grid
        # 2d circular grid
        r, phi = np.mgrid[0:1:ARRAY_LENGTH * 1j, 0:np.pi / 2:ARRAY_LENGTH * 1j]

        # transform to ellipse on the outside, circle on the inside
        x = (ellipse_x / 2 * r + waveguide_throat / 2 * (1 - r)) * np.cos(phi)
        y = (ellipse_y / 2 * r + waveguide_throat / 2 * (1 - r)) * np.sin(phi)

        # compute z profile
        angle_factor = angle_factor / 10000
        z = (ellipse_x / 2 * r / angle_factor) ** (1 / depth_factor)

        throat = np.array(np.column_stack((x[0, 0:ARRAY_LENGTH], y[0, 0:ARRAY_LENGTH],
                                           z[0, 0:ARRAY_LENGTH])))
        ellipse = np.array(np.column_stack((x[ARRAY_LENGTH - 1, 0:ARRAY_LENGTH], y[ARRAY_LENGTH - 1, 0:ARRAY_LENGTH],
                                            z[ARRAY_LENGTH - 1, 0:ARRAY_LENGTH])))
        horizontal_line = np.array(
            np.column_stack((x[0:ARRAY_LENGTH, 0], y[0:ARRAY_LENGTH, 0], z[0:ARRAY_LENGTH, 0])))
        vertical_line = np.array(
            np.column_stack((x[0:ARRAY_LENGTH, ARRAY_LENGTH - 1], y[0:ARRAY_LENGTH, ARRAY_LENGTH - 1],
                             z[0:ARRAY_LENGTH, ARRAY_LENGTH - 1])))
        center_line = np.array(np.column_stack((x[0:ARRAY_LENGTH, 50], y[0:ARRAY_LENGTH, 50], z[0:ARRAY_LENGTH, 50])))

        elliptical_mesh = pv.StructuredGrid(x, y, z)

        return throat, ellipse, center_line, vertical_line, horizontal_line, elliptical_mesh

    def rectangular_calc(self, waveguide_throat, rectangle_x, rectangle_y, depth_factor, angle_factor):
        rectangle_x = rectangle_x / 2
        rectangle_y = rectangle_y / 2
        # waveguide throat line
        phi = np.linspace(0, np.pi / 2, ARRAY_LENGTH)
        x_interior = waveguide_throat / 2 * np.cos(phi)
        y_interior = waveguide_throat / 2 * np.sin(phi)

        # theta is angle where x and y intersect
        theta = np.arctan2(rectangle_y, rectangle_x)
        # find array index which maps to corner of rectangle
        corner_index = (ARRAY_LENGTH * (2 * theta / np.pi)).round().astype(int)
        # construct rectangular coordinate manually
        x_exterior = np.zeros_like(x_interior)
        y_exterior = x_exterior.copy()
        phi_aux = np.linspace(0, theta, corner_index)
        x_exterior[:corner_index] = rectangle_x
        y_exterior[:corner_index] = rectangle_x * np.tan(phi_aux)
        phi_aux = np.linspace(np.pi / 2, theta, ARRAY_LENGTH - corner_index, endpoint=False)[::-1]
        x_exterior[corner_index:] = rectangle_y / np.tan(phi_aux)
        y_exterior[corner_index:] = rectangle_y

        # interpolate between two curves
        r = np.linspace(0, 1, ARRAY_LENGTH)[:, None]  # shape (ARRAY_LENGTH, 1) for broadcasting
        x = x_exterior * r + x_interior * (1 - r)
        y = y_exterior * r + y_interior * (1 - r)

        # compute z profile
        angle_factor = angle_factor / 10000
        z = (rectangle_x / 2 * r / angle_factor) ** (1 / depth_factor)
        # explicitly broadcast to the shape of x and y
        z = np.broadcast_to(z, x.shape)

        # prepare data for export
        throat = np.array(np.column_stack((x[0, 0:ARRAY_LENGTH], y[0, 0:ARRAY_LENGTH],
                                           z[0, 0:ARRAY_LENGTH])))
        x_rectangle = np.array(np.column_stack((x[ARRAY_LENGTH - 1, 0:corner_index],
                                                y[ARRAY_LENGTH - 1, 0:corner_index],
                                                z[ARRAY_LENGTH - 1, 0:corner_index])))

        y_rectangle = np.array(np.column_stack((x[ARRAY_LENGTH - 1, corner_index - 1:ARRAY_LENGTH],
                                                y[ARRAY_LENGTH - 1, corner_index - 1:ARRAY_LENGTH],
                                                z[ARRAY_LENGTH - 1, corner_index - 1:ARRAY_LENGTH])))

        horizontal_line = np.array(
            np.column_stack((x[0:ARRAY_LENGTH, 0], y[0:ARRAY_LENGTH, 0], z[0:ARRAY_LENGTH, 0])))

        vertical_line = np.array(
            np.column_stack((x[0:ARRAY_LENGTH, ARRAY_LENGTH - 1], y[0:ARRAY_LENGTH, ARRAY_LENGTH - 1],
                             z[0:ARRAY_LENGTH, ARRAY_LENGTH - 1])))

        center_line = np.array(
            np.column_stack((x[0:ARRAY_LENGTH, corner_index - 1], y[0:ARRAY_LENGTH, corner_index - 1],
                             z[0:ARRAY_LENGTH, corner_index - 1])))

        y_mid_center_line = np.array(np.column_stack((x[0: ARRAY_LENGTH, int(corner_index / 2)],
                                                      y[0: ARRAY_LENGTH, int(corner_index / 2)],
                                                      z[0: ARRAY_LENGTH, int(corner_index / 2)])))

        x_mid_center_line = np.array(np.column_stack((
            x[0: ARRAY_LENGTH, int(ARRAY_LENGTH - corner_index)],
            y[0: ARRAY_LENGTH, int(ARRAY_LENGTH - corner_index)],
            z[0: ARRAY_LENGTH, int(ARRAY_LENGTH - corner_index)])))

        # Create mesh for 3D viewing
        rectangular_mesh = pv.StructuredGrid(x, y, z)

        return throat, x_rectangle, y_rectangle, center_line, vertical_line, horizontal_line, rectangular_mesh, \
            x_mid_center_line, y_mid_center_line

    def cutoff_frequency(self, coverage_angle, throat_diameter):
        coverage_angle = 0.5 * (coverage_angle * (np.pi / 180))

        throat_radius = (throat_diameter / 2)

        cutoff_freq = (44 * (np.sin(coverage_angle) / (throat_radius / 1000)))

        return cutoff_freq

    def phase_plug_calc(self, plug_dia, dome_dia, plug_offset, ARRAY_LENGTH):
        s, chi = np.mgrid[0:1:ARRAY_LENGTH * 1j, 0:np.pi / 2:ARRAY_LENGTH * 1j]

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
        phaseplug_line = np.array(np.column_stack((x_phaseplug[0:ARRAY_LENGTH, 0], y_phaseplug[0:ARRAY_LENGTH, 0],
                                                   z_phaseplug[0:ARRAY_LENGTH, 0])))

        return phaseplug, phaseplug_line

    def show_warning(self, text, title):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowIcon(QtGui.QIcon("Waveguide_Designer.ico"))
        msg.setText(text)
        msg.setWindowTitle(title)
        msg.exec_()

    def export_obj(self):

        stl_filename = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", '/', "STL (*.stl)")[0]

        if stl_filename == '':
            self.show_warning("No File Was Selected", "Warning")

        else:
            mesh = self.waveguide_whole.triangulate(inplace=True)
            pv.save_meshio(stl_filename, mesh)

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
