import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


# todo: check exist in data dir before adding to tree?

# # Load Kurucz grid points.
# M_H_grid = np.array([-0.1, -0.2, -0.3, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0,
#                      -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0])
# Teff_grid = np.array([3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250,
#                       5500, 5750, 6000, 6250, 6500])
# logg_grid = np.array([4.0, 4.5, 5.0])
# mh_points = []
# Teff_points = []
# logg_points = []
# for M_H in M_H_grid:
#     for Teff in Teff_grid:
#         for logg in logg_grid:
#             mh_points.append(M_H)
#             Teff_points.append(Teff)
#             logg_points.append(logg)
#
# points = np.column_stack((mh_points, Teff_points, logg_points))
# print("kurucz_grid_points shape = {}".format(points.shape))
#
# # Rescale features.
# r_M_H = 1.00
# r_Teff = 607.
# r_logg = 1.54
# points[:, 0] /= r_M_H
# points[:, 1] /= r_Teff
# points[:, 2] /= r_logg
#
# kurucz_tree = KDTree(points, leafsize=10, compact_nodes=True, copy_data=True, balanced_tree=True)
#
# with open('kurucz_tree.pickle', 'wb') as f:
#     pickle.dump(kurucz_tree, f)

with open('kurucz_tree.pickle', 'rb') as f:
    kurucz_tree = pickle.load(f)

r_M_H = 1.00
r_Teff = 607.
r_logg = 1.54

x_M_H = 0.1
x_Teff = 6000
x_logg = 4.5
x = np.array([x_M_H / r_M_H, x_Teff / r_Teff, x_logg / r_logg])

distance, nearest_idx = kurucz_tree.query(x, k=1)
nearest_M_H, nearest_Teff, nearest_logg = kurucz_tree.data[nearest_idx]
print(nearest_M_H, nearest_Teff, nearest_logg)

distances, nearest_idxs = kurucz_tree.query(x, k=3)
nearest_stellar_parameters = kurucz_tree.data[nearest_idxs]
nearest_M_Hs = nearest_stellar_parameters[:, 0] * r_M_H
nearest_Teffs = nearest_stellar_parameters[:, 1] * r_Teff
nearest_loggs = nearest_stellar_parameters[:, 2] * r_logg

print(distances[0] == 0.)
print(distances, nearest_idxs)
print(nearest_M_Hs, nearest_Teffs, nearest_loggs)

