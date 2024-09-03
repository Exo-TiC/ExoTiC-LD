from tqdm import tqdm
import numpy as np
import pkg_resources
import pickle
from pathlib import Path
import os
from glob import glob
from scipy.optimize import curve_fit

from exotic_ld import StellarLimbDarkening
from exotic_ld.ld_laws import (
    linear_ld_law,
    quadratic_ld_law,
    squareroot_ld_law,
    nonlinear_3param_ld_law,
    nonlinear_4param_ld_law,
    kipping_ld_law,
)

# from https://github.com/Exo-TiC/ExoTiC-LD/issues/55
broken_phoenix = [
    (-1.5, 6900, 5.0),
    (-1.5, 6700, 2.5),
    (-1.5, 6700, 4.0),
    (-1.5, 6700, 5.0),
    (-1.5, 6700, 5.5),
    (-1.0, 2600, 0.0),
    (-0.5, 2500, 0.0),
    (0.0, 2500, 5.0)
]


class PrecomputedLimbDarkening:
    """
    Precomputed limb darkening class.

    Similar to the StellarLimbDarkening class, but relies on precomputing I_mu values
    across a given stellar grid for a given supported instrument mode. This enables
    faster on-the-fly coefficient calculation that can be useful when sampling.

    Parameters
    ----------
    ld_model : string
        Choose between 'phoenix', 'kurucz', 'stagger', 'mps1', or 'mps2'.
        kurucz are 1D stellar models, can be referenced as '1D'.
        stagger are 3D stellar models, can be referenced as '3D'.
        mps1 are the MPS-ATLAS set 1 models. mps2 are the MPS-ATLAS
        set 2 models. Note that PrecomputedLimbDarkening does not support
        custom stellar models.
    ld_data_path : string
        Path to exotic-ld-data directory. As of version>=3.2.0 this path
        specifies where stellar and instrument data are automatically
        downloaded and stored. Only the required data is downloaded, and
        if the data has previously been used, then no download is required.
        The directory will be automatically created on the first call.
        It remains an option, and is backwards compatible, to download
        all the data from zenodo and specify the path.
    mode : string
        Instrument mode that defines the throughput.
        Modes supported for Hubble:
            'HST_STIS_G430L', 'HST_STIS_G750L', 'HST_WFC3_G280p1',
            'HST_WFC3_G280n1', 'HST_WFC3_G102', 'HST_WFC3_G141'.
        Modes supported for JWST:
            'JWST_NIRSpec_Prism', 'JWST_NIRSpec_G395H',
            'JWST_NIRSpec_G395M', 'JWST_NIRSpec_G235H',
            'JWST_NIRSpec_G235M', 'JWST_NIRSpec_G140H-f100',
            'JWST_NIRSpec_G140M-f100', 'JWST_NIRSpec_G140H-f070',
            'JWST_NIRSpec_G140M-f070', 'JWST_NIRISS_SOSSo1',
            'JWST_NIRISS_SOSSo2', 'JWST_NIRCam_F322W2',
            'JWST_NIRCam_F444', 'JWST_MIRI_LRS'.
        Modes for photometry:
            'Spitzer_IRAC_Ch1', 'Spitzer_IRAC_Ch2', 'TESS'.
    wavelength_range : array_like, (start, end), optional
        Wavelength range over which to compute the limb-darkening
        coefficients. Wavelengths must be given in angstroms and
        the values must fall within the supported range of the
        corresponding instrument mode. If None, will be set to the
        min/max wavelengths of the throughput.
    custom_wavelengths : array_like, optional
        Wavelengths corresponding to custom_throughput [angstroms].
    custom_throughput : array_like, optional
        Throughputs corresponding to custom_wavelengths.
    interpolate_type : string
        Choose between 'nearest' and 'trilinear'.
    ld_data_version : string
        Version number of the data files. Implemented at 3.2.0, and
        this corresponds to files with no version number appended.
        Recommend not changing this from the default value.
    verbose : int
        Level of printed information during calculation. Default: 1.
        0 (no info), 1 (warnings/downloads), 2 (step-by-step info).
    save_stellar_grid : bool
        Whether to save all of the stellar models when building a
        a cache. Defaults to yes for Kurucz and Stagger models,
        and no for MPS and Phoenix models.

    Examples
    --------
    >>> from exotic_ld import PrecomputedLimbDarkening
    >>> sld = PrecomputedLimbDarkening(
            ld_model="kurucz",
            ld_data_path=EXOTIC_LD_PATH,
            mode="JWST_NIRSpec_Prism",
            wavelength_range=None,
            custom_wavelengths=None,
            custom_throughput=None,
            interpolate_type="nearest",
            ld_data_version="3.2.0",
            verbose=2,
            save_stellar_grid=None
        )
    >>> i_mu1 = sld.get_I_mu(mh=0.1, teff=5240, logg=4.2, interpolate_type="trilinear")
    >>> i_mu2 = sld.get_I_mu(mh=0.1, teff=5280, logg=4.2, interpolate_type="trilinear")

    """

    def __init__(
        self,
        ld_model,
        ld_data_path,
        mode,
        wavelength_range=None,
        custom_wavelengths=None,
        custom_throughput=None,
        interpolate_type="nearest",
        ld_data_version="3.2.0",
        verbose=1,
        save_stellar_grid=None,
    ):
        # general setup
        self.ld_model = ld_model
        self.ld_data_path = ld_data_path
        self.mode = mode
        self.wavelength_range = wavelength_range
        self.custom_wavelengths = custom_wavelengths
        self.custom_throughput = custom_throughput
        self.interpolate_type = interpolate_type
        self.verbose = verbose
        self.save_stellar_grid = save_stellar_grid

        self._r_M_H = 1.00
        self._r_Teff = 607.0
        self._r_logg = 1.54
        self.scale = np.array([self._r_M_H, self._r_Teff, self._r_logg])

        self.ld_data_path = ld_data_path
        self.remote_ld_data_path = "https://www.star.bris.ac.uk/exotic-ld-data"
        self.ld_data_version = ld_data_version
        if self.ld_data_version == "3.2.0":
            self.ld_data_version = ""  # Ensures backwards compatibility.
        if ld_model == "1D":
            self.ld_model = "kurucz"
        elif ld_model == "3D":
            self.ld_model = "stagger"
        else:
            self.ld_model = ld_model

        self._get_grid_mus()
        self._set_wavelengths()
        self._set_save_behavior()
        self._load_tree()
        self._load_cache()

        if self.verbose > 1:
            print("Ready!")

    def __repr__(self):
        return (
            f"PrecomputedLimbDarkening(ld_model={self.ld_model}, "
            f"mode={self.mode}, "
            f"wavelength_range={self.wavelength_range}, "
        )

    def _get_grid_mus(self):
        # see if any stellar spectra from the grid have already been downloaded
        stellar_spectra = glob(
            os.path.join(f"{self.ld_data_path}/{self.ld_model}/" "**", "*.dat"),
            recursive=True,
        )
        # if not, download one representative model to get the mu values
        if len(stellar_spectra) == 0:
            if self.verbose > 1:
                print("downloading one representative model to get the mu values")
            z = StellarLimbDarkening(
                M_H=0.0,
                Teff=5500,
                logg=4.0,
                ld_model=self.ld_model,
                ld_data_path=self.ld_data_path,
                interpolate_type="nearest",
                verbose=self.verbose,
            )
            self.mus = z.mus
        else:
            self.mus = np.loadtxt(stellar_spectra[0], skiprows=1, max_rows=1)

    def _set_wavelengths(self):
        if self.verbose > 1:
            print("setting the wavelength range...")
        if self.mode == "custom":
            assert (self.custom_wavelengths is not None) & (
                self.custom_throughput is not None
            ), "If using a custom mode, custom_wavelengths and custom_throughput are required"
        if self.wavelength_range is None:
            # get the sensitivity info for the mode
            if self.mode != "custom":
                z = StellarLimbDarkening(
                    M_H=0.0,
                    Teff=5500,
                    logg=4.0,
                    ld_model=self.ld_model,
                    ld_data_path=self.ld_data_path,
                    interpolate_type="nearest",
                    verbose=self.verbose,
                )
                sensitivity_wavelengths, sensitivity_throughputs = (
                    z._read_sensitivity_data(mode=self.mode)
                )

                # set the wavelength range to the max and min of the bandpass
                self.wavelength_range = [
                    np.min(sensitivity_wavelengths),
                    np.max(sensitivity_wavelengths),
                ]
            else:
                self.wavelength_range = [
                    np.min(self.custom_wavelengths),
                    np.max(self.custom_wavelengths),
                ]
            if self.verbose > 1:
                print(
                    f"\nNo wavelength range provided, so using the max and min of the provided mode ({self.mode}): {self.wavelength_range} angstrom"
                )
        # create a cache file name that's unique to the grid, mode, and wavelength range
        w = (
            str(int(self.wavelength_range[0]))
            + "_"
            + str(int(self.wavelength_range[1]))
        )
        self.cache_file = (
            f"{self.ld_data_path}/cached_mus/{self.ld_model}_{self.mode}_{w}.npy"
        )

    def _set_save_behavior(self):
        # choose whether to save the downloaded stellar spectra
        # necessary since some grids are huge
        if self.save_stellar_grid is None:
            if self.ld_model == "kurucz":
                self.save_stellar_grid = True
                size = 0.4

            elif self.ld_model == "stagger":
                self.save_stellar_grid = True
                size = 1.7

            elif (self.ld_model == "mps1") | (self.ld_model == "mps2"):
                self.save_stellar_grid = False
                size = 21

            elif self.ld_model == "phoenix":
                self.save_stellar_grid = False
                size = 409

            if self.verbose > 0:
                s1 = "are" if self.save_stellar_grid else "are not"
                s2 = "and save" if self.save_stellar_grid else "but not save"
                print(f"\n{self.ld_model} models {s1} saved by default")
                print(
                    f"proceeding may download {s2} ~{size} GB of data if it's not already present"
                )

    def _load_tree(self):
        # get the KD tree to find all of the MH/teff/logg combinations
        if self.verbose > 1:
            print("\nretrieving KD tree")
        file_path = pkg_resources.resource_filename(
            "grid_build.kd_trees", f"{self.ld_model}_tree.pickle"
        )
        with open(file_path, "rb") as f:
            self.tree = pickle.load(f)

    def _load_cache(self):
        # try to load a pre-saved cache
        try:
            self.precomputed_mus = np.load(self.cache_file)
            if self.verbose > 1:
                print(f"\ncache loaded from: {self.cache_file}")
        # if this grid/mode/wavelength range combo hasn't been computed yet, build its cache
        except FileNotFoundError:
            if self.verbose > 0:
                print(
                    f"\n\nWARNING: pre-computed values not found in {self.cache_file}"
                )
                print("building cache now\n\n")
            self._build_cache()
            self.precomputed_mus = np.load(self.cache_file)

    def _build_cache(self):
        # rescale the leafs to their physical values
        leafs = self.tree.data.copy()
        leafs[:, 1] *= self._r_Teff
        leafs[:, 2] *= self._r_logg

        data = np.zeros((len(leafs), len(self.mus)))
        c = 0
        # loop through every stellar model in the grid, saving the I_mu values
        for model in tqdm(leafs):
            # # skip the broken phoenix models from issue #55
            # if self.ld_model == "phoenix" and tuple(model) in broken_phoenix:
            #     data[c] = np.zeros(len(self.mus))*np.nan

            metal, temp, logg = model

            sld = StellarLimbDarkening(
                M_H=metal,
                Teff=temp,
                logg=logg,
                ld_model=self.ld_model,
                ld_data_path=self.ld_data_path,
                interpolate_type="nearest",  # they should all be exact matches
                custom_wavelengths=None,
                custom_mus=None,
                custom_stellar_model=None,
                ld_data_version=self.ld_data_version,
                verbose=self.verbose,
            )

            sld._integrate_I_mu(
                wavelength_range=self.wavelength_range,
                mode=self.mode,
                custom_wavelengths=self.custom_wavelengths,
                custom_throughput=self.custom_throughput,
            )

            # if you don't want to save the stellar spectra, delete them
            if not self.save_stellar_grid:
                metal_str = str(metal).replace("-0.0", "0.0")
                temp_str = (
                    str(int(round(temp / 50) * 50))
                    if self.ld_model != "stagger"
                    else str(int(temp))
                )
                logg_str = f"{logg:.1f}"
                local_file = (
                    self.ld_data_path
                    + f"/{self.ld_model}/MH{metal_str}/teff{temp_str}/logg{logg_str}/{self.ld_model}_spectra.dat"
                )
                if os.path.isfile(local_file):
                    os.remove(local_file)

            data[c] = sld.I_mu
            c += 1  # don't like the way tqdm looks when using enumerate, so using this instead

        if self.verbose > 1:
            print("saving the completed cache")
        Path(self.ld_data_path + "/cached_mus").mkdir(parents=True, exist_ok=True)
        np.save(self.cache_file, data)
        if self.verbose > 1:
            print(f"cache saved to: {self.cache_file}")

        return data

    def _retrieve_mus(self, mh, teff, logg):
        # get the I_mu values for a given set of stellar parameters
        # the cache index is the same as the KD tree index
        x = np.array([mh, teff, logg]) / self.scale
        _, idx = self.tree.query(x, k=1)
        return self.precomputed_mus[idx]

    def _get_surrounding_grid_cuboid(self, mh, teff, logg):
        """
        Exact copy of StellarGrids._get_surrounding_grid_cuboid(),
        just can take in mh, teff, and logg values instead of
        relying on the class attributes.
        """
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

    def _fit_ld_law(self, I_mu, ld_law_func, mu_min, return_sigmas):
        # exact copy of StellarLimbDarkening._fit_ld_law(),
        # just with I_mu now as an input

        # Truncate mu range to be fitted.
        mu_mask = self.mus >= mu_min

        if np.sum(mu_mask) < 2:
            raise ValueError(
                "mu_min={} set too high, must be >= 2 mu "
                "values remaining.".format(mu_min)
            )

        if not ld_law_func == kipping_ld_law:
            if self.verbose > 1:
                print(
                    "Fitting limb-darkening law to {} I(mu) data points "
                    "where {} <= mu <= 1, with the Levenberg-Marquardt "
                    "algorithm.".format(np.sum(mu_mask), mu_min)
                )

            # Fit limb-darkening law: Levenberg-Marquardt (LM), guess=1.
            popt, pcov = curve_fit(
                ld_law_func, self.mus[mu_mask], I_mu[mu_mask], method="lm"
            )

        else:
            if self.verbose > 1:
                print(
                    "Fitting limb-darkening law to {} I(mu) data points "
                    "where {} <= mu <= 1, with the constrained Trust-Region "
                    "Reflective algorithm.".format(np.sum(mu_mask), mu_min)
                )

            # Fit limb-darkening law: Trust-Region Reflective (TRF), guess=0.5.
            # For the Kipping law, constrain q1, q2 to [0, 1].
            popt, pcov = curve_fit(
                kipping_ld_law,
                self.mus[mu_mask],
                I_mu[mu_mask],
                p0=(0.5, 0.5),
                bounds=((0, 0), (1, 1)),
                method="trf",
            )

        if self.verbose > 1:
            print("Fit done, resulting coefficients are {}.".format(popt))

        if return_sigmas:
            return tuple(popt), tuple(np.sqrt(np.diag(pcov)))
        else:
            return tuple(popt)

    def get_I_mu(self, M_H, Teff, logg, interpolate_type=None):
        """
        Compute the stellar intensity profile for a given set of stellar parameters.

        If doing "nearest" interpolation, this will return the exact same
        values called ._integrate_I_mu() on StellarLimbDarkening object.
        The two approaches diverge when using "trilinear" interpolation
        though: this will interpolate across I_mu profiles, while
        StellarLimbDarkening will interpolate across the stellar spectra
        before integrating to get I_mu. This method does not require
        any file I/O, making it faster than the StellarLimbDarkening

        Parameters
        ----------
        M_H : float
            Stellar metallicity [dex].
        Teff : int
            Stellar effective temperature [kelvin].
        logg : float
            Stellar log(g) [dex].
        interpolate_type : string
            Choose between 'nearest' and 'trilinear'.

        Returns
        -------
        I_mu : array_like
            Intensity profile across mu values associated with the stellar grid.


        """
        if interpolate_type is None:
            interpolate_type = self.interpolate_type

        if interpolate_type == "nearest":

            if self.verbose > 1:
                print("Using interpolation type = nearest.")

            # Find best-matching grid point to input parameters.
            x = np.array([M_H, Teff, logg]) / self.scale
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

            return self._retrieve_mus(M_H, Teff, logg)

        elif interpolate_type == "trilinear":
            if self.verbose > 1:
                print("Using interpolation type = trilinear.")

            vertices = self._get_surrounding_grid_cuboid(M_H, Teff, logg)

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

                return self.get_I_mu(M_H, Teff, logg, interpolate_type="nearest")

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
            xd = (M_H - x0) / (x1 - x0) if x0 != x1 else 0.0
            yd = (Teff - y0) / (y1 - y0) if y0 != y1 else 0.0
            zd = (logg - z0) / (z1 - z0) if z0 != z1 else 0.0

            c00 = c000 * (1 - xd) + c100 * xd
            c01 = c001 * (1 - xd) + c101 * xd
            c10 = c010 * (1 - xd) + c110 * xd
            c11 = c011 * (1 - xd) + c111 * xd

            c0 = c00 * (1 - yd) + c10 * yd
            c1 = c01 * (1 - yd) + c11 * yd

            c = c0 * (1 - zd) + c1 * zd

            return c

    ####################################################################################
    # the limb darkening laws, all copied from StellarLimbDarkening
    # except with fixed wavelengths/mode and free M_H, Teff, logg
    ####################################################################################
    def compute_linear_ld_coeffs(
        self, M_H, Teff, logg, mu_min=0.10, return_sigmas=False
    ):
        """
        Compute the linear limb-darkening coefficients.

        Parameters
        ----------
        M_H : float
            Stellar metallicity [dex].
        Teff : int
            Stellar effective temperature [kelvin].
        logg : float
            Stellar log(g) [dex].
        mu_min : float
            Minimum value of mu to include in the fitting process.
        return_sigmas : boolean
            Return the uncertainties, or standard deviations, of each
            fitted limb-darkening coefficient. Default: False.

        Returns
        -------
        if return_sigmas == False:
            (c1, ) : tuple
                Limb-darkening coefficients for the linear law.
        else:
            ((c1, ), (c1_sigma, )) : tuple of tuples
                Limb-darkening coefficients for the linear law
                and uncertainties on each coefficient.

        """

        I_mu = self.get_I_mu(M_H, Teff, logg)
        return self._fit_ld_law(I_mu, linear_ld_law, mu_min, return_sigmas)

    def compute_quadratic_ld_coeffs(
        self, M_H, Teff, logg, mu_min=0.10, return_sigmas=False
    ):
        """
        Compute the quadratic limb-darkening coefficients.

        Parameters
        ----------
        M_H : float
            Stellar metallicity [dex].
        Teff : int
            Stellar effective temperature [kelvin].
        logg : float
            Stellar log(g) [dex].
        mu_min : float
            Minimum value of mu to include in the fitting process.
        return_sigmas : boolean
            Return the uncertainties, or standard deviations, of each
            fitted limb-darkening coefficient. Default: False.

        Returns
        -------
        if return_sigmas == False:
            (c1, c2) : tuple
                Limb-darkening coefficients for the quadratic law.
        else:
            ((c1, c2), (c1_sigma, c2_sigma)) : tuple of tuples
                Limb-darkening coefficients for the quadratic law
                and uncertainties on each coefficient.

        """
        I_mu = self.get_I_mu(M_H, Teff, logg)
        return self._fit_ld_law(I_mu, quadratic_ld_law, mu_min, return_sigmas)

    def compute_kipping_ld_coeffs(
        self, M_H, Teff, logg, mu_min=0.10, return_sigmas=False
    ):
        """
        Compute the Kipping limb-darkening coefficients. These are based on
        a reparameterisation of the quadratic law as described in
        Kipping 2013, MNRAS, 435, 2152. See equations 15 -- 18:

        u1 = 2 * q1^0.5 * q2
        u2 = q1^0.5 * (1 - 2 * q2)

        or,

        q1 = (u1 + u2)^2,
        q2 = 0.5 * u1 * (u1 + u2)^-1.

        Parameters
        ----------
        M_H : float
            Stellar metallicity [dex].
        Teff : int
            Stellar effective temperature [kelvin].
        logg : float
            Stellar log(g) [dex].
        mu_min : float
            Minimum value of mu to include in the fitting process.
        return_sigmas : boolean
            Return the uncertainties, or standard deviations, of each
            fitted limb-darkening coefficient. Default: False.

        Returns
        -------
        if return_sigmas == False:
            (c1, c2) : tuple
                Limb-darkening coefficients for the Kipping law.
        else:
            ((c1, c2), (c1_sigma, c2_sigma)) : tuple of tuples
                Limb-darkening coefficients for the Kipping law
                and uncertainties on each coefficient.

        """
        I_mu = self.get_I_mu(M_H, Teff, logg)
        return self._fit_ld_law(I_mu, kipping_ld_law, mu_min, return_sigmas)

    def compute_squareroot_ld_coeffs(
        self, M_H, Teff, logg, mu_min=0.10, return_sigmas=False
    ):
        """
        Compute the square root limb-darkening coefficients.

        Parameters
        ----------
        M_H : float
            Stellar metallicity [dex].
        Teff : int
            Stellar effective temperature [kelvin].
        logg : float
            Stellar log(g) [dex].
        mu_min : float
            Minimum value of mu to include in the fitting process.
        return_sigmas : boolean
            Return the uncertainties, or standard deviations, of each
            fitted limb-darkening coefficient. Default: False.

        Returns
        -------
        if return_sigmas == False:
            (c1, c2) : tuple
                Limb-darkening coefficients for the square root law.
        else:
            ((c1, c2), (c1_sigma, c2_sigma)) : tuple of tuples
                Limb-darkening coefficients for the square root law
                and uncertainties on each coefficient.

        """
        I_mu = self.get_I_mu(M_H, Teff, logg)
        return self._fit_ld_law(I_mu, squareroot_ld_law, mu_min, return_sigmas)

    def compute_3_parameter_non_linear_ld_coeffs(
        self, M_H, Teff, logg, mu_min=0.10, return_sigmas=False
    ):
        """
        Compute the three-parameter non-linear limb-darkening coefficients.

        Parameters
        ----------
        M_H : float
            Stellar metallicity [dex].
        Teff : int
            Stellar effective temperature [kelvin].
        logg : float
            Stellar log(g) [dex].
        mu_min : float
            Minimum value of mu to include in the fitting process.
        return_sigmas : boolean
            Return the uncertainties, or standard deviations, of each
            fitted limb-darkening coefficient. Default: False.

        Returns
        -------
        if return_sigmas == False:
            (c1, c2, c3) : tuple
                Limb-darkening coefficients for the three-parameter
                non-linear law.
        else:
            ((c1, c2, c3), (c1_sigma, c2_sigma, c3_sigma)) : tuple of tuples
                Limb-darkening coefficients for the three-parameter
                non-linear law and uncertainties on each coefficient.

        """
        I_mu = self.get_I_mu(M_H, Teff, logg)
        return self._fit_ld_law(I_mu, nonlinear_3param_ld_law, mu_min, return_sigmas)

    def compute_4_parameter_non_linear_ld_coeffs(
        self, M_H, Teff, logg, mu_min=0.10, return_sigmas=False
    ):
        """
        Compute the four-parameter non-linear limb-darkening coefficients.

        Parameters
        ----------
        M_H : float
            Stellar metallicity [dex].
        Teff : int
            Stellar effective temperature [kelvin].
        logg : float
            Stellar log(g) [dex].
        mu_min : float
            Minimum value of mu to include in the fitting process.
        return_sigmas : boolean
            Return the uncertainties, or standard deviations, of each
            fitted limb-darkening coefficient. Default: False.

        Returns
        -------
        if return_sigmas == False:
            (c1, c2, c3, c4) : tuple
                Limb-darkening coefficients for the three-parameter
                non-linear law.
        else:
            ((c1, c2, c3, c4), (c1_sigma, c2_sigma, c3_sigma, c4_sigma)) :
                tuple of tuples
                Limb-darkening coefficients for the four-parameter
                non-linear law and uncertainties on each coefficient.

        """
        I_mu = self.get_I_mu(M_H, Teff, logg)
        return self._fit_ld_law(I_mu, nonlinear_4param_ld_law, mu_min, return_sigmas)

    @staticmethod
    def _verify_local_data(ld_model, ld_data_path):
        """
        Check if you have all of the files downloaded for a given stellar grid.

        Not really necessary since you could download each file one-by-one as needed,
        but after bulk downloading the grids it was good to check that everything is there.
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
