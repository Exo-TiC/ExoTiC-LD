import os
import numpy as np


# Phoenix grid points.
ld_data_path_original = '../../exotic_ld_data_original'
stellar_data_path = os.path.join(ld_data_path_original, 'phoenix_v3_SpecIntFITS')
M_H_grid = np.array([-4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.0, 0.5, 1.0])
Teff_grid = np.concatenate([np.arange(2300, 7000, 100), np.arange(7000, 12000, 200), np.arange(12000, 15001, 500)])
logg_grid = np.arange(0.0, 6.01, 0.5)
mh_points = []
Teff_points = []
logg_points = []
for M_H in M_H_grid:
    for Teff in Teff_grid:
        for logg in logg_grid:
            t_str = "{0:05}".format(Teff)
            lg_str = "{0:+.2f}".format(-logg)
            mh_str = "{0:+.1f}".format(M_H)
            file_name = "lte{}{}{}.PHOENIX-ACES-AGSS-COND-2011-SpecInt.fits".format(t_str, lg_str, mh_str)
            file_path = os.path.join(stellar_data_path, "Z{}".format(mh_str), file_name)

            if os.path.exists(file_path):
                mh_points.append(M_H)
                Teff_points.append(Teff)
                logg_points.append(logg)

points = np.column_stack((mh_points, Teff_points, logg_points))
print("phoenix_grid_points shape = {}".format(points.shape))
np.savetxt("phoenix_grid_points.dat", points, delimiter=" ", header="phoenix MH Teff logg v3.2")

# Kurucz grid points.
M_H_grid = np.array([-0.1, -0.2, -0.3, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0,
                     -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0])
Teff_grid = np.array([3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250,
                      5500, 5750, 6000, 6250, 6500])
logg_grid = np.array([4.0, 4.5, 5.0])
mh_points = []
Teff_points = []
logg_points = []
for M_H in M_H_grid:
    for Teff in Teff_grid:
        for logg in logg_grid:
            mh_points.append(M_H)
            Teff_points.append(Teff)
            logg_points.append(logg)

points = np.column_stack((mh_points, Teff_points, logg_points))
print("kurucz_grid_points shape = {}".format(points.shape))
np.savetxt("kurucz_grid_points.dat", points, delimiter=" ", header="kurucz MH Teff logg v3.2")

# mps1 grid points.
M_H_grid = np.array([-0.1, -0.2, -0.3, -0.4, -0.5, -0.05, -0.6, -0.7, -0.8,
                     -0.9, -0.15, -0.25, -0.35, -0.45, -0.55, -0.65, -0.75,
                     -0.85, -0.95, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6,
                     -1.7, -1.8, -1.9, -2.0, -2.1, -2.2, -2.3, -2.4, -2.5,
                     -3.0, -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.4,
                     0.5, 0.05, 0.6, 0.7, 0.8, 0.9, 0.15, 0.25, 0.35, 0.45,
                     1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
Teff_grid = np.arange(3500, 9050, 100)
logg_grid = np.array([3.0, 3.5, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.0])
mh_points = []
Teff_points = []
logg_points = []
for M_H in M_H_grid:
    for Teff in Teff_grid:
        for logg in logg_grid:
            mh_points.append(M_H)
            Teff_points.append(Teff)
            logg_points.append(logg)

points = np.column_stack((mh_points, Teff_points, logg_points))
print("mps1_grid_points shape = {}".format(points.shape))
np.savetxt("mps1_grid_points.dat", points, delimiter=" ", header="mps-atlas-1 MH Teff logg v3.2")

# mps2 grid points.
M_H_grid = np.array([-0.1, -0.2, -0.3, -0.4, -0.5, -0.05, -0.6, -0.7, -0.8,
                     -0.9, -0.15, -0.25, -0.35, -0.45, -0.55, -0.65, -0.75,
                     -0.85, -0.95, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6,
                     -1.7, -1.8, -1.9, -2.0, -2.1, -2.2, -2.3, -2.4, -2.5,
                     -3.0, -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.4,
                     0.5, 0.05, 0.6, 0.7, 0.8, 0.9, 0.15, 0.25, 0.35, 0.45,
                     1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
Teff_grid = np.arange(3500, 9050, 100)
logg_grid = np.array([3.0, 3.5, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.0])
mh_points = []
Teff_points = []
logg_points = []
for M_H in M_H_grid:
    for Teff in Teff_grid:
        for logg in logg_grid:
            mh_points.append(M_H)
            Teff_points.append(Teff)
            logg_points.append(logg)

points = np.column_stack((mh_points, Teff_points, logg_points))
print("mps2_grid_points shape = {}".format(points.shape))
np.savetxt("mps2_grid_points.dat", points, delimiter=" ", header="mps-atlas-2 MH Teff logg v3.2")

# stagger grid points.
irregular_grid = {4000: {1.5: [-3.0, -2.0, -1.0, 0.0],
                         2.0: [-3.0, -2.0, -1.0, 0.0],
                         2.5: [-3.0, -2.0, -1.0, 0.0]},
                  4500: {1.5: [-3.0, -1.0],
                         2.0: [-3.0, -2.0, -1.0, 0.0],
                         2.5: [-3.0, -2.0, -1.0, 0.0],
                         3.0: [-3.0, -1.0, 0.0],
                         3.5: [-3.0, 0.0],
                         4.0: [-3.0, 0.0],
                         4.5: [0.0],
                         5.0: [0.0]},
                  5000: {2.0: [-3.0, 0.0],
                         2.5: [-3.0, 0.0],
                         3.0: [-3.0, 0.0],
                         3.5: [-3.0, -1.0, 0.0],
                         4.0: [-3.0, -2.0, -1.0, 0.0],
                         4.5: [-3.0, -2.0, -1.0, 0.0],
                         5.0: [-3.0, -2.0, -1.0, 0.0]},
                  5500: {2.5: [-3.0, -2.0],
                         3.0: [-3.0, -2.0, -1.0, 0.0],
                         3.5: [-3.0, -1.0, 0.0],
                         4.0: [-3.0, -2.0, -1.0, 0.0],
                         4.5: [-3.0, -2.0, -1.0, 0.0],
                         5.0: [-3.0, -2.0, -1.0, 0.0]},
                  5777: {4.4: [-3.0, -2.0, -1.0, 0.0]},
                  6000: {3.5: [-3.0, -2.0, -1.0, 0.0],
                         4.0: [-3.0, -2.0, -1.0, 0.0],
                         4.5: [-3.0, -2.0, -1.0, 0.0]},
                  6500: {4.0: [-3.0, -2.0, -1.0, 0.0],
                         4.5: [-3.0, -2.0, -1.0, 0.0]},
                  7000: {4.5: [-3.0, 0.0]}}
mh_points = []
Teff_points = []
logg_points = []
Teff_grid = np.fromiter(irregular_grid.keys(), dtype=int)
for Teff in Teff_grid:
    logg_grid = np.fromiter(irregular_grid[Teff].keys(), dtype=float)
    for logg in logg_grid:
        M_H_grid = np.array(irregular_grid[Teff][logg], dtype=float)
        for M_H in M_H_grid:
            mh_points.append(M_H)
            Teff_points.append(Teff)
            logg_points.append(logg)

points = np.column_stack((mh_points, Teff_points, logg_points))
print("stagger_grid_points shape = {}".format(points.shape))
np.savetxt("stagger_grid_points.dat", points, delimiter=" ", header="stagger MH Teff logg v3.2")
