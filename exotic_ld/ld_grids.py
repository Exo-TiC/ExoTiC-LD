import os
import numpy as np


class StellarGrids(object):
    """
    Stellar grids class.

    Check availability of stellar models in each of the
    various grids supported. Load nearest match or
    interpolate stellar models within a cuboid.

    """

    def __init__(self, M_H, Teff, logg, ld_model, ld_data_path,
                 interpolate_type, verbose):
        self.verbose = verbose

        self.M_H_input = M_H
        self.Teff_input = Teff
        self.logg_input = logg
        self.ld_model = ld_model
        self.ld_data_path = ld_data_path
        self.interpolate_type = interpolate_type

        self._M_H_grid = None
        self._Teff_grid = None
        self._logg_grid = None
        self._irregular_grid = None

    def get_stellar_data(self):
        # Define grid coverage.
        if self.ld_model == "kurucz":
            self._M_H_grid = np.array(
                [-0.1, -0.2, -0.3, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0,
                 -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0])
            self._Teff_grid = np.array(
                [3500, 3750, 4000, 4250, 4500, 4750, 5000,
                 5250, 5500, 5750, 6000, 6250, 6500])
            self._logg_grid = np.array([4.0, 4.5, 5.0])

        elif self.ld_model == "mps1" or self.ld_model == "mps2":
            self._M_H_grid = np.array(
                [-0.1, -0.2, -0.3, -0.4, -0.5, -0.05, -0.6, -0.7, -0.8,
                 -0.9, -0.15, -0.25, -0.35, -0.45, -0.55, -0.65, -0.75,
                 -0.85, -0.95, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6,
                 -1.7, -1.8, -1.9, -2.0, -2.1, -2.2, -2.3, -2.4, -2.5,
                 -3.0, -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.4,
                 0.5, 0.05, 0.6, 0.7, 0.8, 0.9, 0.15, 0.25, 0.35, 0.45,
                 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
            self._Teff_grid = np.arange(3500, 9050, 100)
            self._logg_grid = np.array(
                [3.0, 3.5, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.0])

        elif self.ld_model == "stagger":
            self._irregular_grid =\
                {4000: {1.5: [-3.0, -2.0, -1.0, 0.0],
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

        else:
            raise ValueError("ld_model not recognised.")

        # Get stellar data.
        if not self.ld_model == "stagger":
            if self.M_H_input in self._M_H_grid \
                    and self.Teff_input in self._Teff_grid \
                    and self.logg_input in self._logg_grid:
                # Exact input parameters exist in grid.
                if self.verbose:
                    print("Exact match found.")
                return self._read_in_stellar_model(
                    self.M_H_input, self.Teff_input, self.logg_input)
        else:
            try:
                if self.M_H_input in self._irregular_grid[
                        self.Teff_input][self.logg_input]:
                    if self.verbose:
                        print("Exact match found.")
                    return self._read_in_stellar_model(
                        self.M_H_input, self.Teff_input, self.logg_input)
                else:
                    raise KeyError
            except KeyError as err:
                pass

        if self.interpolate_type == "nearest":
            # Find best-matching grid point to input parameters.
            M_H, Teff, logg = self._get_nearest_grid_point()
            if self.verbose:
                print("Matched nearest with M_H={}, Teff={}, logg={}."
                      .format(M_H, Teff, logg))
            return self._read_in_stellar_model(M_H, Teff, logg)

        elif self.interpolate_type == "trilinear":
            # Trilinear interpolation of cuboid of grid points
            # (x8 vertices) surrounding input parameters.
            x0, x1, y0, y1, z0, z1 = self._get_surrounding_grid_cuboid()
            if self.verbose:
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

    def _read_in_stellar_model(self, M_H, Teff, logg):
        """ Read in the stellar model from disc. """
        file_name = os.path.join(
            self.ld_data_path, self.ld_model,
            "MH{}".format(M_H),
            "teff{}".format(Teff),
            "logg{}".format(logg),
            "{}_spectra.dat".format(self.ld_model))

        try:
            mus = np.loadtxt(file_name, skiprows=1, max_rows=1)
            stellar_data = np.loadtxt(file_name, skiprows=2)
            return stellar_data[:, 0], mus, stellar_data[:, 1:]

        except FileNotFoundError as err:
            raise FileNotFoundError(
                'Model not found for stellar grid={} at path={}.'.format(
                 self.ld_model, file_name))

    def _get_nearest_grid_point(self):
        if not self.ld_model == "stagger":
            # Find nearest grid point for each input parameter.
            match_M_H_idx = np.argmin(np.abs(self._M_H_grid - self.M_H_input))
            match_Teff_idx = np.argmin(np.abs(self._Teff_grid - self.Teff_input))
            match_logg_idx = np.argmin(np.abs(self._logg_grid - self.logg_input))

            match_M_H = self._M_H_grid[match_M_H_idx]
            match_Teff = self._Teff_grid[match_Teff_idx]
            match_logg = self._logg_grid[match_logg_idx]
        else:
            # Find nearest grid point in order of Teff, logg, then M_H.
            Teff_grid = np.fromiter(self._irregular_grid.keys(), dtype=int)
            match_Teff_idx = np.argmin(np.abs(Teff_grid - self.Teff_input))
            match_Teff = Teff_grid[match_Teff_idx]

            logg_grid = np.fromiter(self._irregular_grid[match_Teff].keys(), dtype=float)
            match_logg_idx = np.argmin(np.abs(logg_grid - self.logg_input))
            match_logg = logg_grid[match_logg_idx]

            M_H_grid = np.array(self._irregular_grid[match_Teff][match_logg], dtype=float)
            match_M_H_idx = np.argmin(np.abs(M_H_grid - self.M_H_input))
            match_M_H = M_H_grid[match_M_H_idx]

        return match_M_H, match_Teff, match_logg

    def _get_surrounding_grid_cuboid(self):
        if not self.ld_model == "stagger":
            # Find adjacent grid points for each parameter
            x0, x1 = self._get_adjacent_grid_points(self._M_H_grid, self.M_H_input)
            y0, y1 = self._get_adjacent_grid_points(self._Teff_grid, self.Teff_input)
            z0, z1 = self._get_adjacent_grid_points(self._logg_grid, self.logg_input)
        else:
            # Find adjacent grid point in order of Teff, logg, then M_H.
            # And check if all other points exist in irregular grid.
            Teff_grid = np.fromiter(self._irregular_grid.keys(), dtype=int)
            y0, y1 = self._get_adjacent_grid_points(Teff_grid, self.Teff_input)

            coords = []
            for y_ in [y0, y1]:
                logg_grid = np.fromiter(self._irregular_grid[y_].keys(), dtype=float)
                z0, z1 = self._get_adjacent_grid_points(logg_grid, self.logg_input)
                for z_ in [z0, z1]:
                    M_H_grid = np.array(self._irregular_grid[y_][z_], dtype=float)
                    x0, x1 = self._get_adjacent_grid_points(M_H_grid, self.M_H_input)
                    for x_ in [x0, x1]:
                        coords.append([y_, z_, x_])

            coords = np.array(coords)
            Teff_consistent_below = coords[:4, 0] == y0
            Teff_consistent_above = coords[4:, 0] == y1
            logg_consistent_below = coords[:, 1].reshape(-1, 4)[:, :2].reshape(-1) == z0
            logg_consistent_above = coords[:, 1].reshape(-1, 4)[:, 2:].reshape(-1) == z1
            M_H_consistent_below = coords[::2, 2] == x0
            M_H_consistent_above = coords[1::2, 2] == x1
            consistent = np.concatenate(
                [Teff_consistent_below, Teff_consistent_above,
                 logg_consistent_below, logg_consistent_above,
                 M_H_consistent_below, M_H_consistent_above])

            if np.any(~consistent):
                raise FileNotFoundError(
                    'Insufficient model coverage to interpolate grid={} at '
                    'M_H={}, Teff={}, logg={}.'.format(
                        self.ld_model, self.M_H_input,
                        self.Teff_input, self.logg_input))

        return x0, x1, y0, y1, z0, z1

    def _get_adjacent_grid_points(self, param_grid, param_input):
        # Find nearest grid point above and below input parameter.
        residual = param_grid - param_input
        if 0. in residual:
            match_idx = np.argmin(np.abs(residual))
            return param_grid[match_idx], param_grid[match_idx]

        residual_plus = residual[residual >= 0.]
        residual_minus = residual[residual < 0.]

        if residual_plus.shape[0] == 0:
            # If no grid above, set both to below.
            below_idx = np.argmax(residual_minus)
            below_param = param_grid[below_idx]
            above_param = below_param

        elif residual_minus.shape[0] == 0:
            # If no grid below, set both to above.
            above_idx = np.argmin(residual_plus)
            above_param = param_grid[above_idx]
            below_param = above_param

        else:
            above_idx = np.argmin(residual_plus)
            below_idx = np.argmax(residual_minus)
            above_param = param_grid[residual >= 0.][above_idx]
            below_param = param_grid[residual < 0.][below_idx]

        return below_param, above_param
