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
phase_plug_dia = 20
plug_offset = 5
dome_dia = 28

# now create the actual structured grid
# 2d circular grid
r, phi = np.mgrid[0:1:array_length*1j, 0:np.pi/2:array_length*1j]
s, chi = np.mgrid[0:1:array_length * 1j, 0:np.pi / 2:array_length * 1j]

# transform to ellipse on the outside, circle on the inside
x = (ellipse_x/2 * r + waveguide_throat/2 * (1 - r))*np.cos(phi)
y = (ellipse_y/2 * r + waveguide_throat/2 * (1 - r))*np.sin(phi)
# dome dia if half dome
x_phaseplug = ((phase_plug_dia / 2 * (1-s)) * np.cos(chi))
y_phaseplug = ((phase_plug_dia / 2 * (1-s)) * np.sin(chi))
# compute z profile
angle_factor = angle_factor / 10000
z = (ellipse_x / 2 * r / angle_factor) ** (1 / depth_factor)

alpha_angle = 2 * np.arcsin(phase_plug_dia / dome_dia)
plug_modification = (phase_plug_dia / 2) / (np.tan(alpha_angle/2))


z_phaseplug = np.sqrt(abs((dome_dia / 2)**2 - (x_phaseplug**2) - (y_phaseplug**2))) + plug_offset - plug_modification
# print(y_phaseplug)

plotter = pv.Plotter()
# waveguide_mesh = pv.StructuredGrid(x, y, z)
# merged = waveguide_mesh.reflect((0,1,0))
# waveguide_mesh = waveguide_mesh.merge(merged)
# slice_y = waveguide_mesh.slice(normal='x', origin=(0.1,0,0))
phase_plug = pv.StructuredGrid(x_phaseplug, y_phaseplug, z_phaseplug)
# plotter.add_mesh(slice_y)
# plotter.add_mesh(waveguide_mesh)
# plotter.show_axes()
# plotter.add_mesh_slice(waveguide_mesh, normal='x')
# plotter.camera_position = 'xz'
# plotter.view_xz()
plotter.add_mesh(phase_plug)
plotter.show()

