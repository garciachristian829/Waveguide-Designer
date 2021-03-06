import numpy as np
import matplotlib as mpl

from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

mpl.use("Qt5Agg")
mpl.rcParams["toolbar"] = "None"  # Get rid of toolbar


def xy_waveguide_contour(throat, x_waveguide, ellipse_x):
    x_initial = (throat + (x_waveguide * (ellipse_x - throat))) / 2

    return x_initial


def z_waveguide_contour(x_array, depth_factor, angle_factor, throat):
    angle_factor = angle_factor / 10000

    x_prime = x_array - (throat / 2)

    z = (x_prime / angle_factor) ** (1 / depth_factor)

    return z


def ellipse_contour(a, b):
    a = a / 2
    b = b / 2
    ellipse_steps = np.linspace(0, 0.5 * np.pi, 100)
    x_ellipse_array = np.array([])
    y_ellipse_array = np.array([])
    for h in range(100):
        x_ellipse_array = np.append(x_ellipse_array, a * np.cos(ellipse_steps[h]))
        y_ellipse_array = np.append(y_ellipse_array, b * np.sin(ellipse_steps[h]))
    return x_ellipse_array, y_ellipse_array


def circle_contour(throat):
    throat = (throat / 2)
    circle_steps = np.linspace(0, 0.5 * np.pi, 100)
    x_circle_array = np.array([])
    y_circle_array = np.array([])
    for j in range(100):
        x_circle_array = np.append(x_circle_array, throat * np.cos(circle_steps[j]))
        y_circle_array = np.append(y_circle_array, throat * np.sin(circle_steps[j]))
    return x_circle_array, y_circle_array


def coverage_calc(x_1, y_1, x_2, y_2):
    slope = (y_1 - y_2) / (x_1 - x_2)

    angle = np.degrees(np.arctan(slope))

    coverage_angle = 180 - (angle * 2)

    return coverage_angle


def main_calc(ax, waveguide_throat, ellipse_x, ellipse_y, depth_fact, angle_fact, phase_plug_dia,
              dome_dia, plug_offset):
    # Total steps = 100
    array_length = 100

    xy_steps = np.linspace(0, 1, 100)

    # initialize x, y, z, and zero array
    x_array = np.array([])
    y_array = np.array([])
    z_array = np.array([])
    zero_array = np.zeros([array_length])

    # calculate contour data and add into array
    for i in range(array_length):
        x_array = np.append(x_array, xy_waveguide_contour(waveguide_throat, xy_steps[i], ellipse_x))
        y_array = np.append(y_array, xy_waveguide_contour(waveguide_throat, xy_steps[i], ellipse_y))

        z_array = np.append(z_array, z_waveguide_contour(x_array[i], depth_fact, angle_fact, waveguide_throat))

    # calculate coverage angle
    calc_coverage_angle = coverage_calc(x_array[49], z_array[49], x_array[51], z_array[51])

    # calculate data for ellipse
    x_ellipse_data, y_ellipse_data = ellipse_contour(ellipse_x, ellipse_y)

    # grab last point from z_array and make entire array same value to define height of ellipse/waveguide
    ellipse_height = z_array[array_length - 1]

    ellipse_z = np.full(shape=array_length, fill_value=ellipse_height)

    # Calculate data for throat
    circle_x, circle_y = circle_contour(waveguide_throat)

    # Calculate phase plug data
    x_phaseplug, y_phaseplug = phase_plug_calc(phase_plug_dia, dome_dia, plug_offset)

    # Reshape arrays into 1 column, multiple rows
    x_array = x_array.reshape(-1, 1)
    y_array = y_array.reshape(-1, 1)
    z_array = z_array.reshape(-1, 1)
    ellipse_z = ellipse_z.reshape(-1, 1)
    x_ellipse_data = x_ellipse_data.reshape(-1, 1)
    y_ellipse_data = y_ellipse_data.reshape(-1, 1)
    circle_x = circle_x.reshape(-1, 1)
    circle_y = circle_y.reshape(-1, 1)
    zero_array = zero_array.reshape(-1, 1)
    x_phaseplug = x_phaseplug.reshape(-1, 1)
    y_phaseplug = y_phaseplug.reshape(-1, 1)

    # save arrays to text

    circle_array = np.concatenate(
        (circle_x, circle_y, zero_array), axis=1
    )
    ellipse_array = np.concatenate(
        (x_ellipse_data, y_ellipse_data, ellipse_z), axis=1
    )
    hor_array = np.concatenate(
        (x_array, zero_array, z_array), axis=1
    )
    ver_array = np.concatenate(
        (zero_array, y_array, z_array), axis=1
    )
    phase_plug = np.concatenate(
        (x_phaseplug, zero_array, y_phaseplug), axis=1
    )
    # Begin plotting all arrays
    ax[0].plot(x_array, z_array, "tab:blue")
    ax[0].plot(x_array * -1, z_array, "tab:blue")
    ax[0].plot(x_phaseplug, y_phaseplug, "tab:green")
    ax[0].plot(x_phaseplug * -1, y_phaseplug, "tab:green")
    ax[1].plot(y_array, z_array, "tab:orange")
    ax[1].plot(y_array * -1, z_array, "tab:orange")
    ax[1].plot(x_phaseplug, y_phaseplug, "tab:green")
    ax[1].plot(x_phaseplug * -1, y_phaseplug, "tab:green")

    ax[0].set(title='Horizontal Curve')
    ax[1].set(title='Vertical Curve')

    ax[0].set_aspect('auto')
    ax[1].set_aspect('auto')
    ax[0].axis('equal')
    ax[1].axis('equal')

    plt.tight_layout()

    # Find max of tweeter
    z_max = np.amax(z_array, axis=0)
    z_max = z_max
    text = "Tweeter height is: %1.2f" % z_max[0] + "mm"
    text_box = AnchoredText(text, loc=3)
    ax[0].add_artist(text_box)

    return circle_array, ellipse_array, hor_array, ver_array, calc_coverage_angle, phase_plug


def save_text_data(circle_array, ellipse_array, hor_array, ver_array, phase_plug, save_text):
    if not phase_plug.any():
        np.savetxt(save_text + "/Throat.txt", circle_array, delimiter=" ")
        np.savetxt(save_text + "/ellipse.txt", ellipse_array, delimiter=" ")
        np.savetxt(save_text + "/hor.txt", hor_array, delimiter=" ")
        np.savetxt(save_text + "/ver.txt", ver_array, delimiter=" ")

    else:
        np.savetxt(save_text + "/Throat.txt", circle_array, delimiter=" ")
        np.savetxt(save_text + "/ellipse.txt", ellipse_array, delimiter=" ")
        np.savetxt(save_text + "/hor.txt", hor_array, delimiter=" ")
        np.savetxt(save_text + "/ver.txt", ver_array, delimiter=" ")
        np.savetxt(save_text + "/phase_plug.txt", phase_plug, delimiter=" ")

    return ()


def cutoff_frequency(coverage_angle, throat_diameter):
    coverage_angle = coverage_angle / 2

    throat_radius = (throat_diameter / 2) / 1000

    cutoff_freq = (44 * (np.radians(np.sin(coverage_angle)) / throat_radius)) * (-1)

    return cutoff_freq


def phase_plug_calc(plug_dia, dome_dia, plug_offset):
    plug_dia = plug_dia / 2
    dome_dia = dome_dia / 2
    x_phase_plug_array = np.array([])
    y_phase_plug_array = np.array([])

    if plug_dia == dome_dia:
        circle_steps = np.linspace(0, 0.5 * np.pi, 100)

        for j in range(100):
            x_phase_plug_array = np.append(x_phase_plug_array, dome_dia * np.cos(circle_steps[j]))
            y_phase_plug_array = np.append(y_phase_plug_array, (dome_dia * np.sin(circle_steps[j])) + plug_offset)

    elif plug_dia < dome_dia > 0:
        alpha_angle = (np.pi * 0.5) - np.arcsin(plug_dia / dome_dia)

        circle_steps = np.linspace(np.pi * 0.5, alpha_angle, 100)

        for j in range(100):
            x_phase_plug_array = np.append(x_phase_plug_array, dome_dia * np.cos(circle_steps[j]))
            y_phase_plug_array = np.append(y_phase_plug_array, (dome_dia * np.sin(circle_steps[j])) + plug_offset)

    return x_phase_plug_array, y_phase_plug_array


if __name__ == "__main__":

    # Run with QT5 UI
    import sys
    from Archive.mainGUI import (
        Ui_MainWindow
    )  # from <filename> of the UI python initialization (content not changed)
    from PyQt5.QtCore import pyqtSlot

    # GLUE CODE: deal with matplotlib
    class MplCanvas(FigureCanvasQTAgg):
        def __init__(self, parent=None, width=5, height=4, dpi=100):
            fig = Figure(figsize=(width, height), dpi=dpi)
            self.axes = fig.add_subplot(111)
            super(MplCanvas, self).__init__(fig)


    class Ui_MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
        def __init__(self):
            super(Ui_MainWindow, self).__init__()

        def setupUi(self, Dialog):
            super().setupUi(Dialog)
            self.groupBox_phaseplug.setEnabled(False)

            self.pushButton_generate_waveguide.clicked.connect(self.generate_waveguide)
            self.pushButton_save_button.clicked.connect(self.on_click2)
            self.checkBox_phaseplug.stateChanged.connect(self.check_state, self.checkBox_phaseplug.isChecked())

        @pyqtSlot()
        def generate_waveguide(self):
            # Get Parameters from LineEdits
            throat_diameter = float(self.lineEdit_throat_diameter.text())  # (1)
            angle_factor = float(self.lineEdit_angle_factor.text())  # (3)
            width = float(self.lineEdit_width.text())  # (4)
            height = float(self.lineEdit_height.text())  # (5)
            depth_factor = float(self.lineEdit_depth_factor.text())  # (6)
            # Check if phase plug parameters are blank or have data
            try:
                phase_plug_dia = float(self.lineEdit_plug_diameter.text())
                dome_dia = float(self.lineEdit_dome_diameter.text())
                phase_plug_offset = float(self.lineEdit_plugoffset.text())
            except:
                phase_plug_dia = 0
                dome_dia = 0
                phase_plug_offset = 0

            # GLUE CODE #3: deal with matplotlib
            # overlay widget and widget_2 with two canvas matplotlib can draw into
            self.canvas1 = MplCanvas(self, width=5, height=4, dpi=100)
            self.canvas2 = MplCanvas(self, width=5, height=4, dpi=100)
            ax = [self.canvas1.axes, self.canvas2.axes]

            self.circle_array, self.ellipse_array, self.hor_array, self.ver_array, self.coverage_angle, self.phaseplug = main_calc(
                ax,
                throat_diameter,
                width,
                height,
                depth_factor,
                angle_factor,
                phase_plug_dia,
                dome_dia,
                phase_plug_offset
            )
            cutoff_freq = cutoff_frequency(self.coverage_angle, throat_diameter)

            self.gridLayout_4.addWidget(self.canvas1, 0, 1, 1, 4)
            self.gridLayout_4.addWidget(self.canvas2, 1, 1, 1, 4)

            self.canvas1.draw()
            self.canvas2.draw()

            coverage_angle = str(int(self.coverage_angle))
            cutoff_freq = str(int(cutoff_freq))

            self.lineEdit_coverage_angle.setText(coverage_angle)
            self.lineEdit_cutoff_freq.setText(cutoff_freq)

        def on_click2(self):
            circle_array = self.circle_array
            ellipse_array = self.ellipse_array
            hor_array = self.hor_array
            ver_array = self.ver_array
            phase_plug = self.phaseplug

            save_text = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))

            save_text_data(circle_array, ellipse_array, hor_array, ver_array, phase_plug, save_text)

        def check_state(self, state):

            if state == 0:
                self.groupBox_phaseplug.setEnabled(False)
                self.lineEdit_dome_diameter.clear()
                self.lineEdit_plug_diameter.clear()
                self.lineEdit_plugoffset.clear()
            else:
                self.groupBox_phaseplug.setEnabled(True)


    # MAIN APP
    #
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QMainWindow()
    # Extent class Ui_Dialog with GLUE CODE 1-3
    ui = Ui_MainWindow()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
