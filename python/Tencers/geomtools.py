import numpy as np
import numpy.linalg as la
from scipy.spatial.transform import Rotation
import igl
from matplotlib.tri import Triangulation as mpl_triangulation
import matplotlib.pyplot as plt

#################################################
###     Some geometrical tools for viewers.   ###
#################################################

def cylindrical_trimesh_between(a, b, ra, rb):
	"""Construct a cylindrical triangular mesh between two points a and b, with radius ra and rb"""
	n_div = 3
	r = Rotation.from_euler('z', np.linspace(0, 2 * np.pi, n_div, endpoint=False), degrees=False)
	height = vnorm(b - a)
	top_layer = r.apply(np.array([ra, 0, 0 + height]))
	bottom_layer = r.apply(np.array([rb, 0, 0]))
	verts = np.row_stack((top_layer, bottom_layer))

	# triangles between the two layers
	tris = np.row_stack((
		[[i, (i + 1) % n_div, n_div + (i + 1) % n_div] for i in range(n_div)],
		[[n_div + (i + 1) % n_div, n_div + i, i] for i in range(n_div)],
	))

	assert np.all(tris >= 0)

	# rotate the cylinder from the z axis to the direction of (b - a)
	z_axis = np.array([0., 0., 1.])
	ax = normalized(b - a)
	to_world = Rotation.from_matrix(rotation_between_vectors(z_axis, ax))
	w_verts = to_world.apply(verts) + a
	
	return w_verts, tris.astype(np.uint)

def vnorm(v):
	if v.ndim == 1:
		return np.linalg.norm(v)
	elif v.ndim == 2:
		return np.linalg.norm(v, axis=1)

def normalized(v):
	return v / np.linalg.norm(v)

def rotation_between_vectors(a, b):
	"""
	find the rotation matrix R s.t R a = b. 
	If a, b are not unit vector, they will be normalized first
	"""
	a_n = a / np.linalg.norm(a)
	b_n = b / np.linalg.norm(b)

	if np.allclose(np.cross(a_n, b_n), 0):
		return np.eye(3) * np.sign(np.dot(a_n, b_n))  # if a_n and b_n opposite, inverse

	v = np.cross(a_n, b_n)
	c = np.dot(a_n, b_n)
	s = np.linalg.norm(v)
	kmat = np.array([
		[0, -v[2], v[1]], 
		[v[2], 0, -v[0]], 
		[-v[1], v[0], 0]])

	rot_mat = np.eye(3) + kmat + (kmat @ kmat) * ((1 - c) / (s ** 2))
	return rot_mat