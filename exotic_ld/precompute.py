from tqdm import tqdm
import numpy as np
import pkg_resources
import pickle
from pathlib import Path
import os
from glob import glob

from exotic_ld import StellarLimbDarkening


class PrecomputedLimbDarkening:
    def __init__(
        self, ld_model, ld_data_path, mode, interpolate_type="nearest", verbose=1
    ):
        self.ld_model = ld_model
        self.ld_data_path = ld_data_path
        self.mode = mode
        self.interpolate_type = interpolate_type
        self.verbose = verbose

        self._r_M_H = 1.00
        self._r_Teff = 607.0
        self._r_logg = 1.54
        self.scale = np.array([self._r_M_H, self._r_Teff, self._r_logg])

        stellar_spectra = glob(
            os.path.join(f"{self.ld_data_path}/{self.ld_model}/" "**", "*.dat"),
            recursive=True,
        )
        if len(stellar_spectra) == 0:
            if self.verbose > 1:
                print("downloading one representative model to get the mu values")
            z = StellarLimbDarkening(
                M_H=0.0,
                Teff=5500,
                logg=4.0,
                ld_model=ld_model,
                ld_data_path=ld_data_path,
                interpolate_type="nearest",
                verbose=1,
            )
            self.mus = z.mus
        else:
            self.mus = np.loadtxt(stellar_spectra[0], skiprows=1, max_rows=1)

        self._load_tree()
        self._load_cache()

        if self.verbose > 1:
            print("Ready!")

    def _load_tree(self):
        # get the KD tree to find all of the MH/teff/logg combinations
        if self.verbose > 1:
            print("retrieving and parsing KD tree")
        file_path = pkg_resources.resource_filename(
            "grid_build.kd_trees", f"{self.ld_model}_tree.pickle"
        )
        with open(file_path, "rb") as f:
            self.tree = pickle.load(f)

    def _load_cache(self):
        filename = f"{self.ld_data_path}/cached_mus/{self.ld_model}_{self.mode}.npy"
        try:
            self.precomputed_mus = np.load(filename)
            if self.verbose > 1:
                print(f"cache loaded from: {filename}")
        except FileNotFoundError:
            print(f"pre-computed values not found in {filename}")
            print(
                f"you can build the cache with the `build_cache` method, but take note of the memory requirements of downloading the whole stellar grid"
            )
            raise FileNotFoundError

    @staticmethod
    def _verify_local_data(ld_model, ld_data_path):
        """
        Check if you have all of the files downloaded for a given stellar grid.

        Not really necessary since you could download each file one-by-one as needed, but
        after bulk downloading the grids it was good to check that nothing failed.
        """

        # get the KD-tree associated with that grid
        file_path = pkg_resources.resource_filename(
            "grid_build.kd_trees", f"{ld_model}_tree.pickle"
        )
        with open(file_path, "rb") as f:
            tree = pickle.load(f)

        # rescale the leafs to their physical values
        leafs = tree.data
        leafs[:, 1] *= 607.0
        leafs[:, 2] *= 1.54

        # create the skeleton of a directory
        if ld_model != "stagger":  # stagger has a 5777K model
            paths = [
                f"{ld_model}/MH{i[0]}/teff{int(round(i[1] / 50) * 50)}/logg{i[2]:.1f}"
                for i in leafs
            ]
        else:
            paths = [
                f"{ld_model}/MH{i[0]}/teff{int(i[1])}/logg{i[2]:.1f}" for i in leafs
            ]

        paths = [i.replace("-0.0/", "0.0/") for i in paths]

        local_paths = [
            ld_data_path + "/" + p + f"/{ld_model}_spectra.dat" for p in paths
        ]

        exists = np.array([os.path.isfile(p) for p in local_paths])

        return np.sum(exists) == len(exists)

    @staticmethod
    def build_cache(
        ld_model,
        ld_data_path,
        mode,
        wavelength_range=None,
        custom_wavelengths=None,
        custom_throughput=None,
        verbose=1,
    ):
        if mode == "custom":
            assert (
                wavelength_range is not None
            ), "If using a custom mode, a wavelength range is required"

        # get the KD tree to find all of the MH/teff/logg combinations
        if verbose > 1:
            print("retrieving and parsing KD tree")
        file_path = pkg_resources.resource_filename(
            "grid_build.kd_trees", f"{ld_model}_tree.pickle"
        )
        with open(file_path, "rb") as f:
            tree = pickle.load(f)

        # rescale the leafs to their physical values
        leafs = tree.data
        leafs[:, 1] *= 607.0
        leafs[:, 2] *= 1.54

        if mode != "stagger":
            models = [
                (
                    str(i[0]).replace("-0.0", "0.0"),
                    str(int(round(i[1] / 50) * 50)),
                    f"{i[2]:.1f}",
                )
                for i in leafs
            ]
        else:
            models = [
                (str(i[0]).replace("-0.0", "0.0"), str(int(i[1])), f"{i[2]:.1f})")
                for i in leafs
            ]

        data = []  # not sure the mu shape at first, so leave as a list
        for model in tqdm(models):
            metal_str, temp_str, logg_str = model

            sld = StellarLimbDarkening(
                M_H=float(metal_str),
                Teff=float(temp_str),
                logg=float(logg_str),
                ld_model=ld_model,
                ld_data_path=ld_data_path,
                interpolate_type="nearest",
                verbose=1,
            )

            if wavelength_range is None:
                sensitivity_wavelengths, sensitivity_throughputs = (
                    sld._read_sensitivity_data(mode=mode)
                )
                wavelength_range = [
                    np.min(sensitivity_wavelengths),
                    np.max(sensitivity_wavelengths),
                ]
                if verbose > 1:
                    print(
                        f"No wavelength range provided, so using the max and min of the provided mode ({mode}): [{wavelength_range[0]}, {wavelength_range[1]}] angstrom"
                    )

            sld._integrate_I_mu(
                wavelength_range=wavelength_range,
                mode=mode,
                custom_wavelengths=custom_wavelengths,
                custom_throughput=custom_throughput,
            )

            data.append(sld.I_mu)
        data = np.array(data)

        if verbose > 1:
            print("saving the completed cache")
        Path(ld_data_path + "/cached_mus").mkdir(parents=True, exist_ok=True)
        np.save(f"{ld_data_path}/cached_mus/{ld_model}_{mode}.npy", data)
        if verbose > 1:
            print(f"cache saved to: {ld_data_path}/cached_mus/{ld_model}_{mode}.npy")

        return data

    def _retrieve_mus(self, mh, teff, logg):
        x = np.array([mh, teff, logg]) / self.scale
        _, idx = self.tree.query(x, k=1)
        return self.precomputed_mus[idx]

    def _get_surrounding_grid_cuboid(self, mh, teff, logg):
        x = np.array([mh, teff, logg]) / self.scale

        # Search scaled radius = 1. from target point; returned arrays are
        # sorted by distance.
        distances, near_idxs = self.tree.query(
            x, k=len(self.tree.data), distance_upper_bound=1.0
        )
        found_idxs = distances != np.inf
        distances = distances[found_idxs]
        near_idxs = near_idxs[found_idxs]
        near_points = self.tree.data[near_idxs]

        if len(distances) == 0:
            # No nearby points.
            return None

        if distances[0] == 0.0:
            # Exact match found.
            return x[0], x[0], x[1], x[1], x[2], x[2]

        # Now, look for the smallest bounding cuboid. (1) select the first vertex,
        # (2) search for an opposite vertex, one that spans the target in all 3
        # axes, (3) check the remaining 6 vertices exist.

        # Trial points, closest first, as the first vertex.
        for f_idx, first_vertex_idx in enumerate(near_idxs[:-1]):
            first_vertex = self.tree.data[first_vertex_idx]

            # Trial points, only checking those further away, as the opposite vertex.
            for o_idx, opposite_vertex_idx in enumerate(near_idxs[f_idx + 1 :]):
                opposite_vertex = self.tree.data[opposite_vertex_idx]

                # Is this pair of points opposite the target point.
                is_opposite = True
                for i_dim in range(3):
                    if not (
                        first_vertex[i_dim] <= x[i_dim] <= opposite_vertex[i_dim]
                    ) and not (
                        opposite_vertex[i_dim] <= x[i_dim] <= first_vertex[i_dim]
                    ):
                        is_opposite = False
                        break

                if is_opposite:
                    # Check if the remaining 6 vertices exist.
                    remaining_vertices = np.array(
                        [
                            [first_vertex[0], opposite_vertex[1], first_vertex[2]],
                            [first_vertex[0], first_vertex[1], opposite_vertex[2]],
                            [first_vertex[0], opposite_vertex[1], opposite_vertex[2]],
                            [opposite_vertex[0], first_vertex[1], opposite_vertex[2]],
                            [opposite_vertex[0], opposite_vertex[1], first_vertex[2]],
                            [opposite_vertex[0], first_vertex[1], first_vertex[2]],
                        ]
                    )

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

    def get_I_mu(self, mh, teff, logg, interpolate_type=None, guardrails=False):
        if interpolate_type is None:
            interpolate_type = self.interpolate_type

        if interpolate_type == "nearest":

            if ~guardrails:
                return self._retrieve_mus(mh, teff, logg)

            if self.verbose > 1:
                print("Using interpolation type = nearest.")

            # Find best-matching grid point to input parameters.
            x = np.array([mh, teff, logg]) / self.scale
            distance, nearest_idx = self.tree.query(x, k=1)
            nearest_M_H, nearest_Teff, nearest_logg = self.tree.data[nearest_idx]

            # Rescaling.
            nearest_M_H *= self._r_M_H
            nearest_Teff *= self._r_Teff
            nearest_logg *= self._r_logg

            if self.verbose > 0:
                if distance > 1.0:  # Equiv. rescaled units[1.0 dex, 607 K, 1.54 dex].
                    print(
                        "Warning: the closest matching stellar model is far "
                        "from your input at M_H={}, Teff={}, logg={}.".format(
                            nearest_M_H, nearest_Teff, nearest_logg
                        )
                    )
            if self.verbose > 1:
                if distance == 0.0:
                    print(
                        "Exact match found with M_H={}, Teff={}, logg={}.".format(
                            nearest_M_H, nearest_Teff, nearest_logg
                        )
                    )
                else:
                    print(
                        "Matched nearest with M_H={}, Teff={}, logg={}.".format(
                            nearest_M_H, nearest_Teff, nearest_logg
                        )
                    )

            return self._retrieve_mus(mh, teff, logg)

        elif interpolate_type == "trilinear":
            if self.verbose > 1:
                print("Using interpolation type = trilinear.")

            vertices = self._get_surrounding_grid_cuboid(mh, teff, logg)

            if vertices is None:
                if self.verbose > 0:
                    print(
                        "Warning: insufficient model coverage to interpolate grid={} "
                        "at M_H={}, Teff={}, logg={}. Falling back to nearest point.".format(
                            self.ld_model,
                            self.M_H_input,
                            self.Teff_input,
                            self.logg_input,
                        )
                    )

                return self.get_I_mu(mh, teff, logg, interpolate_type="nearest")

            # Rescaling.
            x0, x1, y0, y1, z0, z1 = vertices
            x0 *= self._r_M_H
            x1 *= self._r_M_H
            y0 *= self._r_Teff
            y1 *= self._r_Teff
            z0 *= self._r_logg
            z1 *= self._r_logg

            if self.verbose > 1:
                print(
                    "Trilinear interpolation within M_H={}-{}, Teff={}-{},"
                    " logg={}-{}.".format(x0, x1, y0, y1, z0, z1)
                )

            c000 = self._retrieve_mus(x0, y0, z0)
            c001 = self._retrieve_mus(x0, y0, z1)
            c010 = self._retrieve_mus(x0, y1, z0)
            c100 = self._retrieve_mus(x1, y0, z0)
            c011 = self._retrieve_mus(x0, y1, z1)
            c101 = self._retrieve_mus(x1, y0, z1)
            c110 = self._retrieve_mus(x1, y1, z0)
            c111 = self._retrieve_mus(x1, y1, z1)

            # Compute trilinear interpolation.
            xd = (mh - x0) / (x1 - x0) if x0 != x1 else 0.0
            yd = (teff - y0) / (y1 - y0) if y0 != y1 else 0.0
            zd = (logg - z0) / (z1 - z0) if z0 != z1 else 0.0

            c00 = c000 * (1 - xd) + c100 * xd
            c01 = c001 * (1 - xd) + c101 * xd
            c10 = c010 * (1 - xd) + c110 * xd
            c11 = c011 * (1 - xd) + c111 * xd

            c0 = c00 * (1 - yd) + c10 * yd
            c1 = c01 * (1 - yd) + c11 * yd

            c = c0 * (1 - zd) + c1 * zd

            return c
