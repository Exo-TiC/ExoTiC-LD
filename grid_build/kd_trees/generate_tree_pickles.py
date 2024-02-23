import os
import pickle
import numpy as np
from scipy.spatial import KDTree


def generate_stellar_model_points(M_H_grid=None, Teff_grid=None, logg_grid=None,
                                  irregular_grid=None, ld_model=None,
                                  original_stellar_data_path=None):
    mh_points = []
    Teff_points = []
    logg_points = []
    if irregular_grid is None:
        for M_H in M_H_grid:
            for Teff in Teff_grid:
                for logg in logg_grid:
                    if ld_model == "phoenix":
                        # Missing seemingly random data, so check.
                        t_str = "{0:05}".format(Teff)
                        lg_str = "{0:+.2f}".format(-logg)
                        mh_str = "{0:+.1f}".format(M_H)
                        file_name = "lte{}{}{}.PHOENIX-ACES-AGSS-COND-2011-SpecInt.fits"\
                                    .format(t_str, lg_str, mh_str)
                        file_path = os.path.join(original_stellar_data_path,
                                                 "phoenix_v3_SpecIntFITS",
                                                 "Z{}".format(mh_str), file_name)
                        if not os.path.exists(file_path):
                            continue

                    mh_points.append(M_H)
                    Teff_points.append(Teff)
                    logg_points.append(logg)
    else:
        Teff_grid = np.fromiter(irregular_grid.keys(), dtype=int)
        for Teff in Teff_grid:
            logg_grid = np.fromiter(irregular_grid[Teff].keys(), dtype=float)
            for logg in logg_grid:
                M_H_grid = np.array(irregular_grid[Teff][logg], dtype=float)
                for M_H in M_H_grid:
                    mh_points.append(M_H)
                    Teff_points.append(Teff)
                    logg_points.append(logg)

    _points = np.column_stack((mh_points, Teff_points, logg_points))
    print("{}_grid_points shape = {}".format(ld_model, _points.shape))

    return _points


# Stellar grid points.
sgp = {"phoenix": {"M_H_grid": np.array([-4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.0, 0.5, 1.0]),
                   "Teff_grid": np.concatenate([np.arange(2300, 7000, 100), np.arange(7000, 12000, 200), np.arange(12000, 15001, 500)]),
                   "logg_grid": np.arange(0.0, 6.01, 0.5)},
       "kurucz": {"M_H_grid": np.array([-0.1, -0.2, -0.3, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0]),
                  "Teff_grid": np.array([3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500]),
                  "logg_grid": np.array([4.0, 4.5, 5.0])},
       "mps1": {"M_H_grid": np.array([-0.1, -0.2, -0.3, -0.4, -0.5, -0.05, -0.6, -0.7, -0.8, -0.9, -0.15, -0.25, -0.35, -0.45, -0.55, -0.65, -0.75, -0.85, -0.95, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, -1.9, -2.0, -2.1, -2.2, -2.3, -2.4, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.05, 0.6, 0.7, 0.8, 0.9, 0.15, 0.25, 0.35, 0.45, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]),
                "Teff_grid": np.arange(3500, 9050, 100),
                "logg_grid": np.array([3.0, 3.5, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.0])},
       "mps2": {"M_H_grid": np.array([-0.1, -0.2, -0.3, -0.4, -0.5, -0.05, -0.6, -0.7, -0.8, -0.9, -0.15, -0.25, -0.35, -0.45, -0.55, -0.65, -0.75, -0.85, -0.95, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, -1.9, -2.0, -2.1, -2.2, -2.3, -2.4, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.05, 0.6, 0.7, 0.8, 0.9, 0.15, 0.25, 0.35, 0.45, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]),
                "Teff_grid": np.arange(3500, 9050, 100),
                "logg_grid": np.array([3.0, 3.5, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.0])},
       "stagger": {"irregular_grid": {4000: {1.5: [-3.0, -2.0, -1.0, 0.0], 2.0: [-3.0, -2.0, -1.0, 0.0], 2.5: [-3.0, -2.0, -1.0, 0.0]},
                                      4500: {1.5: [-3.0, -1.0], 2.0: [-3.0, -2.0, -1.0, 0.0], 2.5: [-3.0, -2.0, -1.0, 0.0], 3.0: [-3.0, -1.0, 0.0], 3.5: [-3.0, 0.0], 4.0: [-3.0, 0.0], 4.5: [0.0], 5.0: [0.0]},
                                      5000: {2.0: [-3.0, 0.0], 2.5: [-3.0, 0.0], 3.0: [-3.0, 0.0], 3.5: [-3.0, -1.0, 0.0], 4.0: [-3.0, -2.0, -1.0, 0.0], 4.5: [-3.0, -2.0, -1.0, 0.0], 5.0: [-3.0, -2.0, -1.0, 0.0]},
                                      5500: {2.5: [-3.0, -2.0], 3.0: [-3.0, -2.0, -1.0, 0.0], 3.5: [-3.0, -1.0, 0.0], 4.0: [-3.0, -2.0, -1.0, 0.0], 4.5: [-3.0, -2.0, -1.0, 0.0], 5.0: [-3.0, -2.0, -1.0, 0.0]},
                                      5777: {4.4: [-3.0, -2.0, -1.0, 0.0]},
                                      6000: {3.5: [-3.0, -2.0, -1.0, 0.0], 4.0: [-3.0, -2.0, -1.0, 0.0], 4.5: [-3.0, -2.0, -1.0, 0.0]},
                                      6500: {4.0: [-3.0, -2.0, -1.0, 0.0], 4.5: [-3.0, -2.0, -1.0, 0.0]},
                                      7000: {4.5: [-3.0, 0.0]}}}
       }

# Scaling parameters; see feature_sensitivity.py analysis.
r_M_H = 1.00
r_Teff = 607.
r_logg = 1.54

# Overwrite pickles?
overwrite_trees = False

# Iterate stellar grids loading points, scaling, and saving as kd-trees.
for ld_model_key, ld_model_grids in sgp.items():
    if ld_model_key == "stagger":
        points = generate_stellar_model_points(
            irregular_grid=ld_model_grids["irregular_grid"],
            ld_model=ld_model_key)
    else:
        points = generate_stellar_model_points(
            M_H_grid=ld_model_grids["M_H_grid"],
            Teff_grid=ld_model_grids["Teff_grid"],
            logg_grid=ld_model_grids["logg_grid"],
            ld_model=ld_model_key,
            original_stellar_data_path="../../exotic_ld_data_original")

    # Scale features.
    points[:, 0] /= r_M_H
    points[:, 1] /= r_Teff
    points[:, 2] /= r_logg

    kd_tree = KDTree(points, leafsize=10, compact_nodes=True, copy_data=True, balanced_tree=True)
    kd_tree_path = '{}_tree.pickle'.format(ld_model_key)
    if os.path.exists(kd_tree_path) and not overwrite_trees:
        print("{} already exists, not overwriting.".format(kd_tree_path))
    else:
        with open(kd_tree_path, 'wb') as f:
            pickle.dump(kd_tree, f)
        print("Created {}.".format(kd_tree_path))
