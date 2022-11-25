import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.special import roots_legendre

from exotic_ld.ld_grids import StellarGrids
from exotic_ld.ld_laws import linear_ld_law, quadratic_ld_law, \
    squareroot_ld_law, nonlinear_3param_ld_law, nonlinear_4param_ld_law


class StellarLimbDarkening(object):
    """
    Stellar limb darkening class.

    Compute the limb darkening coefficients for either 1D or 3D
    stellar models. Limb darkening coefficients are available for
    linear, quadratic, 3-parameter, and 4-parameter laws.

    Parameters
    ----------
    M_H : float
        Stellar metallicity [dex].
    Teff : int
        Stellar effective temperature [kelvin].
    logg : float
        Stellar log(g) [dex].
    ld_model : string, '1D' or '3D'
        Use the 1D or 3D stellar models. Default '1D'.
    ld_data_path : string
        Path to ExoTiC-LD_data directory downloaded from Zenodo. These
        data include the stellar models and instrument throughputs. See
        the docs for further details.


    Methods
    -------
    compute_linear_ld_coeffs()
    compute_quadratic_ld_coeffs()
    compute_3_parameter_non_linear_ld_coeffs()
    compute_4_parameter_non_linear_ld_coeffs()

    Examples
    --------
    >>> from exotic_ld import StellarLimbDarkening
    >>> sld = StellarLimbDarkening(
            M_H=0.1, Teff=6045, logg=4.2, ld_model='1D',
            ld_data_path='path/to/ExoTiC-LD_data')
    >>> c1, c2 = sld.compute_quadratic_ld_coeffs(
            wavelength_range=np.array([20000., 30000.]),
            mode='JWST_NIRSpec_prism')

    """

    def __init__(self, M_H=None, Teff=None, logg=None, ld_model="1D",
                 ld_data_path="", interpolate_type="nearest",
                 custom_wavelengths=None, custom_mus=None,
                 custom_stellar_model=None, verbose=False):
        self.verbose = verbose

        # Stellar input parameters.
        self.M_H_input = float(M_H)
        self.Teff_input = int(Teff)
        self.logg_input = float(logg)
        self.interpolate_type = interpolate_type
        if self.verbose:
            print("Input stellar parameters are M_H={}, Teff={}, logg={}."
                  .format(self.M_H_input, self.Teff_input, self.logg_input))

        # Set stellar grid.
        self.ld_data_path = ld_data_path
        if ld_model == '1D':
            self.ld_model = 'kurucz'
        elif ld_model == '3D':
            self.ld_model = 'stagger'
        else:
            self.ld_model = ld_model

        # Load/build stellar model.
        self.stellar_wavelengths = None
        self.mus = None
        self.stellar_intensities = None
        self.I_mu = None
        if self.ld_model == "custom":
            self.stellar_wavelengths = custom_wavelengths
            self.mus = custom_mus
            self.stellar_intensities = custom_stellar_model
            if self.verbose:
                print("Using custom stellar model.")
        else:
            self._load_stellar_model()
        self._check_stellar_model()

    def __repr__(self):
        return 'Stellar limb darkening: {} models.'.format(self.ld_model)

    def compute_linear_ld_coeffs(self, wavelength_range, mode,
                                 custom_wavelengths=None,
                                 custom_throughput=None,
                                 mu_min=0.10, return_sigmas=False):
        """
        Compute the linear limb-darkening coefficients.

        Parameters
        ----------
        wavelength_range : array_like, (start, end)
            Wavelength range over which to compute the limb-darkening
            coefficients. Wavelengths must be given in angstroms and
            the values must fall within the supported range of the
            corresponding instrument mode.
        mode : string
            Instrument mode which defines the throughput.
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
            Alternatively, use 'custom' mode. In this case the custom
            wavelength and custom throughput must also be specified.
        custom_wavelengths : array_like, optional
            Wavelengths corresponding to custom_throughput [angstroms].
        custom_throughput : array_like, optional
            Throughputs corresponding to custom_wavelengths.

        Returns
        -------
        (c1, ) : tuple
            Limb-darkening coefficients for the linear law.

        """
        # Compute I(mu) for a given response function.
        self._integrate_I_mu(wavelength_range, mode,
                             custom_wavelengths, custom_throughput)

        # Fit limb-darkening law.
        return self._fit_ld_law(linear_ld_law, mu_min, return_sigmas)

    def compute_quadratic_ld_coeffs(self, wavelength_range, mode,
                                    custom_wavelengths=None,
                                    custom_throughput=None,
                                    mu_min=0.10, return_sigmas=False):
        """
        Compute the quadratic limb-darkening coefficients.

        Parameters
        ----------
        wavelength_range : array_like, (start, end)
            Wavelength range over which to compute the limb-darkening
            coefficients. Wavelengths must be given in angstroms and
            the values must fall within the supported range of the
            corresponding instrument mode.
        mode : string
            Instrument mode which defines the throughput.
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
            Alternatively, use 'custom' mode. In this case the custom
            wavelength and custom throughput must also be specified.
        custom_wavelengths : array_like, optional
            Wavelengths corresponding to custom_throughput [angstroms].
        custom_throughput : array_like, optional
            Throughputs corresponding to custom_wavelengths.

        Returns
        -------
        (c1, c2) : tuple
            Limb-darkening coefficients for the quadratic law.

        """
        # Compute I(mu) for a given response function.
        self._integrate_I_mu(wavelength_range, mode,
                             custom_wavelengths, custom_throughput)

        # Fit limb-darkening law.
        return self._fit_ld_law(quadratic_ld_law, mu_min, return_sigmas)

    def compute_squareroot_ld_coeffs(self, wavelength_range, mode,
                                     custom_wavelengths=None,
                                     custom_throughput=None,
                                     mu_min=0.10, return_sigmas=False):
        """
        Compute the square root limb-darkening coefficients.

        Parameters
        ----------
        wavelength_range : array_like, (start, end)
            Wavelength range over which to compute the limb-darkening
            coefficients. Wavelengths must be given in angstroms and
            the values must fall within the supported range of the
            corresponding instrument mode.
        mode : string
            Instrument mode which defines the throughput.
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
            Alternatively, use 'custom' mode. In this case the custom
            wavelength and custom throughput must also be specified.
        custom_wavelengths : array_like, optional
            Wavelengths corresponding to custom_throughput [angstroms].
        custom_throughput : array_like, optional
            Throughputs corresponding to custom_wavelengths.

        Returns
        -------
        (c1, c2) : tuple
            Limb-darkening coefficients for the square root law.

        """
        # Compute I(mu) for a given response function.
        self._integrate_I_mu(wavelength_range, mode,
                             custom_wavelengths, custom_throughput)

        # Fit limb-darkening law.
        return self._fit_ld_law(squareroot_ld_law, mu_min, return_sigmas)

    def compute_3_parameter_non_linear_ld_coeffs(self, wavelength_range, mode,
                                                 custom_wavelengths=None,
                                                 custom_throughput=None,
                                                 mu_min=0.10, return_sigmas=False):
        """
        Compute the three-parameter non-linear limb-darkening coefficients.

        Parameters
        ----------
        wavelength_range : array_like, (start, end)
            Wavelength range over which to compute the limb-darkening
            coefficients. Wavelengths must be given in angstroms and
            the values must fall within the supported range of the
            corresponding instrument mode.
        mode : string
            Instrument mode which defines the throughput.
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
            Alternatively, use 'custom' mode. In this case the custom
            wavelength and custom throughput must also be specified.
        custom_wavelengths : array_like, optional
            Wavelengths corresponding to custom_throughput [angstroms].
        custom_throughput : array_like, optional
            Throughputs corresponding to custom_wavelengths.

        Returns
        -------
        (c1, c2, c3) : tuple
            Limb-darkening coefficients for the three-parameter
            non-linear law.

        """
        # Compute I(mu) for a given response function.
        self._integrate_I_mu(wavelength_range, mode,
                             custom_wavelengths, custom_throughput)

        # Fit limb-darkening law.
        return self._fit_ld_law(nonlinear_3param_ld_law, mu_min, return_sigmas)

    def compute_4_parameter_non_linear_ld_coeffs(self, wavelength_range, mode,
                                                 custom_wavelengths=None,
                                                 custom_throughput=None,
                                                 mu_min=0.10, return_sigmas=False):
        """
        Compute the four-parameter non-linear limb-darkening coefficients.

        Parameters
        ----------
        wavelength_range : array_like, (start, end)
            Wavelength range over which to compute the limb-darkening
            coefficients. Wavelengths must be given in angstroms and
            the values must fall within the supported range of the
            corresponding instrument mode.
        mode : string
            Instrument mode which defines the throughput.
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
            Alternatively, use 'custom' mode. In this case the custom
            wavelength and custom throughput must also be specified.
        custom_wavelengths : array_like, optional
            Wavelengths corresponding to custom_throughput [angstroms].
        custom_throughput : array_like, optional
            Throughputs corresponding to custom_wavelengths.

        Returns
        -------
        (c1, c2, c3, c4) : tuple
            Limb-darkening coefficients for the four-parameter
            non-linear law.

        """
        # Compute I(mu) for a given response function.
        self._integrate_I_mu(wavelength_range, mode,
                             custom_wavelengths, custom_throughput)

        # Fit limb-darkening law.
        return self._fit_ld_law(nonlinear_4param_ld_law, mu_min, return_sigmas)

    def _load_stellar_model(self):
        if self.verbose:
            print("Loading stellar model from {} grid.".format(self.ld_model))

        sg = StellarGrids(self.M_H_input, self.Teff_input,
                          self.logg_input, self.ld_model, self.ld_data_path,
                          self.interpolate_type, self.verbose)
        self.stellar_wavelengths, self.mus, self.stellar_intensities = \
            sg.get_stellar_data()

        if self.verbose:
            print("Stellar model loaded.")

    def _check_stellar_model(self):
        if not self.stellar_intensities.ndim == 2:
            raise ValueError('Stellar intensities must be 2D, with shape '
                             '(n wavelengths, n mu values)')
        if not self.stellar_wavelengths.shape[0] == \
               self.stellar_intensities.shape[0]:
            raise ValueError('Stellar wavelengths must have the same shape as '
                             'stellar_intensities.shape[0].')
        if not self.mus.shape[0] == self.stellar_intensities.shape[1]:
            raise ValueError('mu values must have the same shape as '
                             'stellar_intensities.shape[1].')

    def _read_sensitivity_data(self, mode):
        sensitivity_file_path = os.path.join(
            self.ld_data_path,
            'Sensitivity_files/{}_throughput.csv'.format(mode))
        if not os.path.exists(sensitivity_file_path):
            raise FileNotFoundError(
                'Sensitivity_file not found mode={} at path={}.'.format(
                 mode, sensitivity_file_path))

        sensitivity_data = pd.read_csv(sensitivity_file_path)
        sensitivity_wavelengths = sensitivity_data['wave'].values
        sensitivity_throughputs = sensitivity_data['tp'].values

        return sensitivity_wavelengths, sensitivity_throughputs

    def _integrate_I_mu(self, wavelength_range, mode, custom_wavelengths,
                        custom_throughput):
        if mode == 'custom':
            # Custom throughput provided.
            s_wavelengths = custom_wavelengths
            s_throughputs = custom_throughput
        else:
            # Read in mode specific throughput.
            s_wavelengths, s_throughputs = self._read_sensitivity_data(mode)

        # Select wavelength range.
        s_mask = np.logical_and(wavelength_range[0] < s_wavelengths,
                                s_wavelengths < wavelength_range[1])
        s_wvs = s_wavelengths[s_mask]
        s_thp = s_throughputs[s_mask]

        i_mask = np.logical_and(wavelength_range[0] < self.stellar_wavelengths,
                                self.stellar_wavelengths < wavelength_range[1])
        i_wvs = self.stellar_wavelengths[i_mask]
        i_int = self.stellar_intensities[i_mask]

        # Ready sensitivity interpolator.
        s_interp_func = interp1d(s_wvs, s_thp, kind='linear',
                                 bounds_error=False, fill_value=0.)

        # Pre-compute Gauss-legendre roots and rescale to lims.
        roots, weights = roots_legendre(500)
        a = wavelength_range[0]
        b = wavelength_range[1]
        t = (b - a) / 2 * roots + (a + b) / 2

        # Iterate mu values computing intensity.
        self.I_mu = np.zeros(i_int.shape[1])
        for mu_idx in range(self.mus.shape[0]):

            # Ready intensity interpolator.
            i_interp_func = interp1d(i_wvs, i_int[:, mu_idx], kind='linear',
                                     bounds_error=False, fill_value=0.)

            def integrand(_lambda):
                return s_interp_func(_lambda) * i_interp_func(_lambda)

            # Approximate integral.
            self.I_mu[mu_idx] = (b - a) / 2. * integrand(t).dot(weights)

        # Set I(mu=1) = 1.
        self.I_mu /= self.I_mu[0]

    def _fit_ld_law(self, ld_law_func, mu_min, return_sigmas):
        # Truncate mu range to be fitted.
        mu_mask = self.mus >= mu_min

        # Fit limb-darkening law: levenberg marquardt, guess default=1.
        popt, pcov = curve_fit(ld_law_func,
                               self.mus[mu_mask],
                               self.I_mu[mu_mask],
                               method='lm')

        if return_sigmas:
            return tuple(popt), tuple(np.sqrt(np.diag(pcov)))
        else:
            return tuple(popt)
