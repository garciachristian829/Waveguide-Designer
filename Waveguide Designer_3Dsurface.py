import numpy as np
import pyvista as pv

# parameters for the waveguide
# diameter of the inner circle
waveguide_throat = 30
# axes of the outer ellipse
ellipse_x = 180
ellipse_y = 120
# shape parameters for the z profile
depth_factor = 2.5
angle_factor = 40
# number of grid points in radial and angular direction
array_length = 100

# quarter circle interior line
phi = np.linspace(0, np.pi/2, array_length)
x_interior = waveguide_throat / 2 * np.cos(phi)
y_interior = waveguide_throat / 2 * np.sin(phi)

# theta is angle where x and y intersect
theta = np.arctan2(ellipse_y, ellipse_x)
# find array index which maps to the corner of the rectangle
corner_index = (array_length * (2*theta/np.pi)).round().astype(int)
# construct rectangular coordinates manually
x_exterior = np.zeros_like(x_interior)
y_exterior = x_exterior.copy()
phi_aux = np.linspace(0, theta, corner_index)
x_exterior[:corner_index] = ellipse_x
y_exterior[:corner_index] = ellipse_x * np.tan(phi_aux)
phi_aux = np.linspace(np.pi/2, theta, array_length - corner_index, endpoint=False)[::-1]  # mind the reverse!
x_exterior[corner_index:] = ellipse_y / np.tan(phi_aux)
y_exterior[corner_index:] = ellipse_y

# interpolate between two curves
r = np.linspace(0, 1, array_length)[:, None]  # shape (array_length, 1) for broadcasting
x = x_exterior * r + x_interior * (1 - r)
y = y_exterior * r + y_interior * (1 - r)

# debugging: plot interior and exterior
exterior_points = np.array([
    x_exterior,
    y_exterior,
    np.zeros_like(x_exterior),
]).T
interior_points = np.array([
    x_interior,
    y_interior,
    np.zeros_like(x_interior),
]).T
plotter = pv.Plotter()
plotter.add_mesh(pv.wrap(exterior_points))
plotter.add_mesh(pv.wrap(interior_points))
plotter.show()

# compute z profile
angle_factor = angle_factor / 10000
z = (ellipse_x / 2 * r / angle_factor) ** (1 / depth_factor)
# explicitly broadcast to the shape of x and y
z = np.broadcast_to(z, x.shape)

plotter = pv.Plotter()

waveguide_mesh = pv.StructuredGrid(x, y, z)
plotter.add_mesh(waveguide_mesh, style='wireframe')
plotter.show()
