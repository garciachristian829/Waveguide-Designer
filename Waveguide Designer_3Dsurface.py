import numpy as np
import pyvista as pv

# parameters for the waveguide
# diameter of the inner circle
waveguide_throat = 30
# axes of the outer ellipse
ellipse_x = 250
ellipse_y = 150
# shape parameters for the z profile
depth_factor = 4
angle_factor = 40
# number of grid points in radial and angular direction
array_length = 100

phase_plug = 0
phase_plug_dia = 0
plug_offset = 0
# now create the actual structured grid
# 2d circular grid
r, phi = np.mgrid[0:1:array_length*1j, 0:np.pi/2:array_length*1j]

# transform to ellipse on the outside, circle on the inside
x = (ellipse_x/2 * r + waveguide_throat/2 * (1 - r))*np.cos(phi)
y = (ellipse_y/2 * r + waveguide_throat/2 * (1 - r))*np.sin(phi)

# compute z profile
angle_factor = angle_factor / 10000
z = (ellipse_x / 2 * r / angle_factor) ** (1 / depth_factor)

waveguide = pv.StructuredGrid(x, y, z)
# waveguide.plot()

throat = np.array([np.column_stack((x[0, 0:array_length], y[0, 0:array_length], z[0, 0:array_length]))])
ellipse = np.array([np.column_stack((x[array_length-1, 0:array_length], y[array_length-1, 0:array_length],
                                     z[array_length-1, 0:array_length]))])
horizontal_line = np.array([np.column_stack((x[0:array_length, 0], y[0:array_length, 0], z[0:array_length, 0]))])
vertical_line = np.array([np.column_stack((x[0:array_length, 0], y[0:array_length, 0], z[0:array_length, 0]))])
center_line = np.array([np.column_stack((x[0:array_length, 50], y[0:array_length, 50],
                                         z[0:array_length, 50]))])

