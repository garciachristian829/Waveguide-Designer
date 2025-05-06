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

    # Patch: ensure no zero spacing in z_array
    dz = np.diff(z_array)
    dz[dz == 0] = 1e-9  # or a small epsilon
    # dz = np.where(dz == 0, np.finfo(float).eps, dz)

    z_array = np.cumsum(np.insert(dz, 0, 0)) + z_array[0]

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

        # Define buttons and checkbox state check
        self.pushButton_generate_waveguide.clicked.connect(self.generate_waveguide)
        self.pushButton_save_button.clicked.connect(self.on_click2)
        self.groupBox_phaseplug.toggled.connect(lambda _: None)

        self.actionSave_Waveguide_Parameters.triggered.connect(self.parameters_save)
        self.actionLoad_Waveguide_Parameters.triggered.connect(self.parameters_load)
        self.actionSave_Comsol_Parameters.triggered.connect(self.comsol_parameters)
        self.actionExport_OBJ.triggered.connect(self.export_obj)

        # Set validators
        for le in [self.lineEdit_throat_diameter, self.lineEdit_width, self.lineEdit_height,
                   self.lineEdit_depth, self.lineEdit_plugoffset, self.lineEdit_dome_diameter,
                   self.lineEdit_plug_diameter, self.lineEdit_hor_cov, self.lineEdit_ver_cov]:
            le.setValidator(QtGui.QDoubleValidator(notation=QtGui.QDoubleValidator.StandardNotation))

        self.show()

    def generate_waveguide(self):
        # Clear plotter each time to plot new waveguides
        self.plotter.clear()

        # Get Parameters from LineEdits in Waveguide Groupbox
        throat_diameter = float(self.lineEdit_throat_diameter.text())  # (1)
        depth = float(self.lineEdit_depth.text())  # (2)
        width = float(self.lineEdit_width.text())  # (3)
        height = float(self.lineEdit_height.text())  # (4)
        horizontal_cov = float(self.lineEdit_hor_cov.text())  # (5)
        vertical_cov = float(self.lineEdit_ver_cov.text())

        if self.radioButton_elliptical.isChecked():
            self.circle_array, self.ellipse_array, self.center_array, self.ver_array, self.hor_array, \
                waveguide_mesh \
                = self.elliptical_calc(throat_diameter, width, height, depth, horizontal_cov, vertical_cov)


        elif self.radioButton_rectangular.isChecked():
            self.circle_array, self.x_rectangle, self.y_rectangle, self.center_array, self.ver_array, self.hor_array, \
                waveguide_mesh, self.x_midline, self.y_midline \
                = self.rectangular_calc(throat_diameter, width, height, horizontal_cov, vertical_cov, depth)

        # Calculate zmax of waveguide
        z_max = round(np.max(self.ver_array[:, 2]), 2)

        waveguide_mesh = mirror_mesh(waveguide_mesh)

        # Select scalars to plot "cmap" colors without needing matplotlib
        waveguide_mesh['Data'] = waveguide_mesh.points[:, 2]

        self.waveguide_whole = waveguide_mesh

        if not (self.groupBox_phaseplug.isChecked()):
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
                              str(self.lineEdit_depth.text()) + "\n",
                              str(self.lineEdit_hor_cov.text()) + "\n",
                              str(self.lineEdit_ver_cov.text()) + "\n",
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

            # parameters = open(parameter_file, "r")
            with open(parameter_file, "r") as f:
                content = f.read().splitlines()

            # content = parameters.read().splitlines()

            self.lineEdit_throat_diameter.setText(content[0])
            self.lineEdit_width.setText(content[1])
            self.lineEdit_height.setText(content[2])
            self.lineEdit_depth.setText(content[3])
            self.lineEdit_hor_cov.setText(content[4])
            self.lineEdit_ver_cov.setText(content[5])
            self.checkBox_phaseplug.setChecked(bool(str_to_bool(content[6])))
            self.radioButton_elliptical.setChecked(bool(str_to_bool(content[7])))
            self.radioButton_rectangular.setChecked(bool(str_to_bool(content[8])))

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

    def elliptical_calc(self, throat_diam_mm, mouth_width_mm, mouth_height_mm,
                        depth_mm, hor_cov_deg, ver_cov_deg):
        """
        Elliptical waveguide calculation with enforced tangency and correct dimensions.
        Returns throat, ellipse, centerline, vertical/horizontal cross-sections, and the full mesh.
        """

        # Throat
        throat_radius = throat_diam_mm / 2.0

        # Target mouth radius (used for profile, later rescaled)
        max_mouth_radius = max(mouth_width_mm, mouth_height_mm) / 2.0

        # Use elliptical coverage formula to determine profile bend
        hor_r = np.tan(np.radians(hor_cov_deg / 2))
        ver_r = np.tan(np.radians(ver_cov_deg / 2))
        # Derived from major axis of elliptical wavefront
        major_angle_rad = 2 * np.arctan(np.sqrt(hor_r ** 2 + ver_r ** 2))
        start_angle_deg = np.degrees(major_angle_rad) / 3.0  # matches Excel logic

        start_angle_rad = np.radians(start_angle_deg)
        end_angle_rad = np.radians(90.0)
        exponent = 1.3

        # Generate profile curve (dr/dz) with angle blend
        r_vals = [throat_radius]
        z_vals = [0.0]
        total_steps = ARRAY_LENGTH
        for i in range(1, total_steps):
            t = i / (total_steps - 1)
            angle = start_angle_rad + (end_angle_rad - start_angle_rad) * (t ** exponent)
            ds = 1.0  # use unit steps first, rescale z later
            dz = ds * np.cos(angle)
            dr = ds * np.sin(angle)
            z_vals.append(z_vals[-1] + dz)
            r_vals.append(r_vals[-1] + dr)

        # Convert to arrays
        r_vals = np.array(r_vals)
        z_vals = np.array(z_vals)

        # Rescale profile to match user-defined depth
        actual_depth = z_vals[-1]
        z_vals *= (depth_mm / actual_depth)

        # Rescale r_vals so the final horizontal/vertical reaches target mouth width/height
        # Compute radial delta excluding the throat
        r_delta = r_vals - r_vals[0]
        final_r_delta = r_delta[-1]
        target_r_delta_x = (mouth_width_mm / 2.0) - throat_radius
        target_r_delta_y = (mouth_height_mm / 2.0) - throat_radius

        r_scaled_x = throat_radius + r_delta * (target_r_delta_x / final_r_delta)
        r_scaled_y = throat_radius + r_delta * (target_r_delta_y / final_r_delta)

        # Create elliptical cross-section at each z
        theta = np.linspace(0, np.pi / 2, ARRAY_LENGTH)
        r_grid, theta_grid = np.meshgrid(r_vals, theta, indexing='ij')

        x = r_scaled_x[:, None] * np.cos(theta_grid)
        y = r_scaled_y[:, None] * np.sin(theta_grid)
        z = np.tile(z_vals[:, np.newaxis], (1, ARRAY_LENGTH))

        # Key outputs
        throat = np.column_stack((x[0], y[0], z[0]))
        ellipse = np.column_stack((x[-1], y[-1], z[-1]))
        center_line = np.column_stack((x[:, ARRAY_LENGTH // 2], y[:, ARRAY_LENGTH // 2], z[:, ARRAY_LENGTH // 2]))
        vertical_line = np.column_stack((x[:, -1], y[:, -1], z[:, -1]))
        horizontal_line = np.column_stack((x[:, 0], y[:, 0], z[:, 0]))
        mesh = pv.StructuredGrid(x, y, z)

        return throat, ellipse, center_line, vertical_line, horizontal_line, mesh

    def rectangular_calc(self, throat_diam_mm, mouth_width_mm, mouth_height_mm,
                         hor_cov_deg, ver_cov_deg, depth_mm):
        """
        Rectangular waveguide with circular throat, user-defined depth, and smooth curvature.
        """

        throat_radius = throat_diam_mm / 2.0
        half_width = mouth_width_mm / 2.0
        half_height = mouth_height_mm / 2.0

        # Determine bend profile from elliptical coverage logic
        hor_r = np.tan(np.radians(hor_cov_deg / 2))
        ver_r = np.tan(np.radians(ver_cov_deg / 2))
        major_angle_rad = 2 * np.arctan(np.sqrt(hor_r ** 2 + ver_r ** 2))
        start_angle_deg = np.degrees(major_angle_rad) / 3.0
        start_angle_rad = np.radians(start_angle_deg)
        end_angle_rad = np.radians(90.0)
        exponent = 1.3

        # Generate profile curve (r, z)
        r_vals = [throat_radius]
        z_vals = [0.0]
        for i in range(1, ARRAY_LENGTH):
            t = i / (ARRAY_LENGTH - 1)
            angle = start_angle_rad + (end_angle_rad - start_angle_rad) * (t ** exponent)
            dz = np.cos(angle)
            dr = np.sin(angle)
            z_vals.append(z_vals[-1] + dz)
            r_vals.append(r_vals[-1] + dr)

        r_vals = np.array(r_vals)
        z_vals = np.array(z_vals)
        z_vals *= (depth_mm / z_vals[-1])

        # Rescale r to match mouth dimensions
        r_delta = r_vals - throat_radius
        final_r_delta = r_delta[-1]
        target_r_delta_x = half_width - throat_radius
        target_r_delta_y = half_height - throat_radius

        r_scaled_x = throat_radius + r_delta * (target_r_delta_x / final_r_delta)
        r_scaled_y = throat_radius + r_delta * (target_r_delta_y / final_r_delta)

        # Create elliptical to rectangular transition
        phi = np.linspace(0, np.pi / 2, ARRAY_LENGTH)
        x = np.zeros((ARRAY_LENGTH, ARRAY_LENGTH))
        y = np.zeros_like(x)
        z = np.tile(z_vals[:, np.newaxis], (1, ARRAY_LENGTH))

        for i in range(ARRAY_LENGTH):
            for j in range(ARRAY_LENGTH):
                t = i / (ARRAY_LENGTH - 1)
                angle = phi[j]

                x_circ = throat_radius * np.cos(angle)
                y_circ = throat_radius * np.sin(angle)

                if angle <= np.arctan2(half_height, half_width):
                    x_rect = r_scaled_x[i]
                    y_rect = r_scaled_x[i] * np.tan(angle)
                else:
                    x_rect = r_scaled_y[i] / np.tan(angle)
                    y_rect = r_scaled_y[i]

                x[i, j] = (1 - t) * x_circ + t * x_rect
                y[i, j] = (1 - t) * y_circ + t * y_rect

        # Key outputs
        corner_idx = int(np.round(ARRAY_LENGTH * (2 * np.arctan2(half_height, half_width) / np.pi)))

        throat = np.column_stack((x[0], y[0], z[0]))
        x_rect = np.column_stack((x[-1, :corner_idx], y[-1, :corner_idx], z[-1, :corner_idx]))
        y_rect = np.column_stack((x[-1, corner_idx - 1:], y[-1, corner_idx - 1:], z[-1, corner_idx - 1:]))
        horizontal = np.column_stack((x[:, 0], y[:, 0], z[:, 0]))
        vertical = np.column_stack((x[:, -1], y[:, -1], z[:, -1]))
        center = np.column_stack((x[:, corner_idx - 1], y[:, corner_idx - 1], z[:, corner_idx - 1]))
        x_mid = np.column_stack((x[:, int(ARRAY_LENGTH - corner_idx)],
                                 y[:, int(ARRAY_LENGTH - corner_idx)],
                                 z[:, int(ARRAY_LENGTH - corner_idx)]))
        y_mid = np.column_stack((x[:, int(corner_idx / 2)],
                                 y[:, int(corner_idx / 2)],
                                 z[:, int(corner_idx / 2)]))

        mesh = pv.StructuredGrid(x, y, z)

        return throat, x_rect, y_rect, center, vertical, horizontal, mesh, x_mid, y_mid

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
