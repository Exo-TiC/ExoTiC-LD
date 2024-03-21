import os
import pickle
import numpy as np
import pkg_resources
from scipy.spatial import KDTree

from exotic_ld.ld_requests import download


class StellarGrids(object):
    """
    Stellar grids class.

    Load/download stellar models via kd-tree structure, select nearest
    models by scaled features, and return the stellar data [wvs, mus, Is].

    """

    def __init__(self, M_H, Teff, logg, ld_model, ld_data_path, remote_ld_data_path,
                 ld_data_version, interpolate_type, verbose):
        self.verbose = verbose
        self.M_H_input = M_H
        self.Teff_input = Teff
        self.logg_input = logg
        self.ld_model = ld_model
        self.ld_data_path = ld_data_path
        self.remote_ld_data_path = remote_ld_data_path
        self.interpolate_type = interpolate_type

        # KD-tree of stellar models, scaled(M_H, Teff, logg).
        self._stellar_kd_tree = None
        self._ld_data_version = ld_data_version
        self._get_stellar_model_kd_tree()

        # Scaling parameters.
        self._r_M_H = 1.00
        self._r_Teff = 607.
        self._r_logg = 1.54
        self.x = np.array([self.M_H_input / self._r_M_H,
                           self.Teff_input / self._r_Teff,
                           self.logg_input / self._r_logg])

    def _get_stellar_model_kd_tree(self):
        tree_path = pkg_resources.resource_filename(
            "grid_build.kd_trees", "{}_tree{}.pickle".format(
                self.ld_model, self._ld_data_version))
        try:
            with open(tree_path, "rb") as f:
                self._stellar_kd_tree = pickle.load(f)
        except FileNotFoundError as err:
            raise ValueError("ld_model not recognised.")
        except pickle.UnpicklingError as err:
            raise pickle.UnpicklingError("Failed loading stellar model pickle, "
                                         "check python version is supported.")

    def get_stellar_data(self):
        """ Return stellar data ready for computing limb darkening. """
        if self.interpolate_type == "nearest":
            if self.verbose > 1:
                print("Using interpolation type = nearest.")

            # Find best-matching grid point to input parameters.
            distance, nearest_idx = self._stellar_kd_tree.query(self.x, k=1)
            nearest_M_H, nearest_Teff, nearest_logg = \
                self._stellar_kd_tree.data[nearest_idx]

            # Rescaling.
            nearest_M_H *= self._r_M_H
            nearest_Teff *= self._r_Teff
            nearest_logg *= self._r_logg

            if self.verbose > 0:
                if distance > 1.0:  # Equiv. rescaled units[1.0 dex, 607 K, 1.54 dex].
                    print("Warning: the closest matching stellar model is far "
                          "from your input at M_H={}, Teff={}, logg={}."
                          .format(nearest_M_H, nearest_Teff, nearest_logg))
            if self.verbose > 1:
                if distance == 0.:
                    print("Exact match found with M_H={}, Teff={}, logg={}."
                          .format(nearest_M_H, nearest_Teff, nearest_logg))
                else:
                    print("Matched nearest with M_H={}, Teff={}, logg={}."
                          .format(nearest_M_H, nearest_Teff, nearest_logg))

            return self._read_in_stellar_model(nearest_M_H, nearest_Teff, nearest_logg)

        elif self.interpolate_type == "trilinear":
            if self.verbose > 1:
                print("Using interpolation type = trilinear.")

            # Trilinear interpolation of cuboid of grid points
            # (x8 vertices) surrounding input parameters.
            vertices = self._get_surrounding_grid_cuboid()

            if vertices is None:
                if self.verbose > 0:
                    print("Warning: insufficient model coverage to interpolate grid={} "
                          "at M_H={}, Teff={}, logg={}. Falling back to nearest point."
                          .format(self.ld_model, self.M_H_input,
                                  self.Teff_input, self.logg_input))

                self.interpolate_type = "nearest"
                return self.get_stellar_data()

            # Rescaling.
            x0, x1, y0, y1, z0, z1 = vertices
            x0 *= self._r_M_H
            x1 *= self._r_M_H
            y0 *= self._r_Teff
            y1 *= self._r_Teff
            z0 *= self._r_logg
            z1 *= self._r_logg

            if self.verbose > 1:
                print("Trilinear interpolation within M_H={}-{}, Teff={}-{},"
                      " logg={}-{}.".format(x0, x1, y0, y1, z0, z1))

            wvs, mus, c000 = self._read_in_stellar_model(x0, y0, z0)
            _, _, c001 = self._read_in_stellar_model(x0, y0, z1)
            _, _, c010 = self._read_in_stellar_model(x0, y1, z0)
            _, _, c100 = self._read_in_stellar_model(x1, y0, z0)
            _, _, c011 = self._read_in_stellar_model(x0, y1, z1)
            _, _, c101 = self._read_in_stellar_model(x1, y0, z1)
            _, _, c110 = self._read_in_stellar_model(x1, y1, z0)
            _, _, c111 = self._read_in_stellar_model(x1, y1, z1)

            # Compute trilinear interpolation.
            xd = (self.M_H_input - x0) / (x1 - x0) if x0 != x1 else 0.
            yd = (self.Teff_input - y0) / (y1 - y0) if y0 != y1 else 0.
            zd = (self.logg_input - z0) / (z1 - z0) if z0 != z1 else 0.

            c00 = c000 * (1 - xd) + c100 * xd
            del c000, c100
            c01 = c001 * (1 - xd) + c101 * xd
            del c001, c101
            c10 = c010 * (1 - xd) + c110 * xd
            del c010, c110
            c11 = c011 * (1 - xd) + c111 * xd
            del c011, c111

            c0 = c00 * (1 - yd) + c10 * yd
            del c00, c10
            c1 = c01 * (1 - yd) + c11 * yd
            del c01, c11

            c = c0 * (1 - zd) + c1 * zd

            return wvs, mus, c

        else:
            raise ValueError("interpolate_type not recognised.")

    def _get_surrounding_grid_cuboid(self):
        # Search scaled radius = 1. from target point; returned arrays are
        # sorted by distance.
        distances, near_idxs = self._stellar_kd_tree.query(
            self.x, k=len(self._stellar_kd_tree.data), distance_upper_bound=1.0)
        found_idxs = distances != np.inf
        distances = distances[found_idxs]
        near_idxs = near_idxs[found_idxs]
        near_points = self._stellar_kd_tree.data[near_idxs]

        if len(distances) == 0:
            # No nearby points.
            return None

        if distances[0] == 0.:
            # Exact match found.
            return self.x[0], self.x[0], self.x[1], self.x[1], self.x[2], self.x[2]

        # Now, look for the smallest bounding cuboid. (1) select the first vertex,
        # (2) search for an opposite vertex, one that spans the target in all 3
        # axes, (3) check the remaining 6 vertices exist.

        # Trial points, closest first, as the first vertex.
        for f_idx, first_vertex_idx in enumerate(near_idxs[:-1]):
            first_vertex = self._stellar_kd_tree.data[first_vertex_idx]

            # Trial points, only checking those further away, as the opposite vertex.
            for o_idx, opposite_vertex_idx in enumerate(near_idxs[f_idx + 1:]):
                opposite_vertex = self._stellar_kd_tree.data[opposite_vertex_idx]

                # Is this pair of points opposite the target point.
                is_opposite = True
                for i_dim in range(3):
                    if not (first_vertex[i_dim] <= self.x[i_dim] <=
                            opposite_vertex[i_dim]) \
                            and not (opposite_vertex[i_dim] <= self.x[i_dim] <=
                                     first_vertex[i_dim]):
                        is_opposite = False
                        break

                if is_opposite:
                    # Check if the remaining 6 vertices exist.
                    remaining_vertices = np.array(
                        [[first_vertex[0], opposite_vertex[1], first_vertex[2]],
                         [first_vertex[0], first_vertex[1], opposite_vertex[2]],
                         [first_vertex[0], opposite_vertex[1], opposite_vertex[2]],
                         [opposite_vertex[0], first_vertex[1], opposite_vertex[2]],
                         [opposite_vertex[0], opposite_vertex[1], first_vertex[2]],
                         [opposite_vertex[0], first_vertex[1], first_vertex[2]]])

                    exists_cuboid = True
                    for rv in remaining_vertices:
                        if not np.any(np.all(near_points == rv, axis=1)):
                            exists_cuboid = False
                            break

                    if exists_cuboid:
                        # Order vertices by position.
                        if first_vertex[0] < opposite_vertex[0]:
                            x0 = first_vertex[0]
                            x1 = opposite_vertex[0]
                        else:
                            x0 = opposite_vertex[0]
                            x1 = first_vertex[0]

                        if first_vertex[1] < opposite_vertex[1]:
                            y0 = first_vertex[1]
                            y1 = opposite_vertex[1]
                        else:
                            y0 = opposite_vertex[1]
                            y1 = first_vertex[1]

                        if first_vertex[2] < opposite_vertex[2]:
                            z0 = first_vertex[2]
                            z1 = opposite_vertex[2]
                        else:
                            z0 = opposite_vertex[2]
                            z1 = first_vertex[2]

                        return x0, x1, y0, y1, z0, z1

        return None

    def _read_in_stellar_model(self, M_H, Teff, logg):
        M_H = 0.0 if M_H == -0.0 else M_H  # Make zeros not negative.

        local_file_path = os.path.join(
            self.ld_data_path, self.ld_model,
            "MH{}".format(str(round(M_H, 2))),
            "teff{}".format(int(round(Teff))),
            "logg{}".format(str(round(logg, 1))),
            "{}_spectra{}.dat".format(self.ld_model, self._ld_data_version))
        remote_file_path = os.path.join(
            self.remote_ld_data_path, self.ld_model,
            "MH{}".format(str(round(M_H, 2))),
            "teff{}".format(int(round(Teff))),
            "logg{}".format(str(round(logg, 1))),
            "{}_spectra{}.dat".format(self.ld_model, self._ld_data_version))

        # Check if exists locally.
        if not os.path.exists(local_file_path):
            download(remote_file_path, local_file_path, self.verbose)
            if self.verbose > 1:
                print("Downloaded {}.".format(local_file_path))

        mus = np.loadtxt(local_file_path, skiprows=1, max_rows=1)
        stellar_data = np.loadtxt(local_file_path, skiprows=2)
        return stellar_data[:, 0], mus, stellar_data[:, 1:]
