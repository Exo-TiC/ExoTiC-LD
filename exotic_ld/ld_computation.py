import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.special import roots_legendre

from exotic_ld.ld_grids import StellarGrids
from exotic_ld.ld_requests import download
from exotic_ld.ld_laws import linear_ld_law, quadratic_ld_law, \
    squareroot_ld_law, nonlinear_3param_ld_law, nonlinear_4param_ld_law, \
    kipping_ld_law


class StellarLimbDarkening(object):
    """
    Stellar limb darkening class.

    Compute the limb-darkening coefficients for a specified
    stellar model, instrument throughput, and limb-darkening law.

    Parameters
    ----------
    M_H : float
        Stellar metallicity [dex].
    Teff : int
        Stellar effective temperature [kelvin].
    logg : float
        Stellar log(g) [dex].
    ld_model : string
        Choose between 'phoenix', 'kurucz', 'stagger', 'mps1', 'mps2', or 'custom'.
        kurucz are 1D stellar models, can be referenced as '1D'.
        stagger are 3D stellar models, can be referenced as '3D'.
        mps1 are the MPS-ATLAS set 1 models. mps2 are the MPS-ATLAS
        set 2 models. If custom, must also provide custom_wavelengths,
        custom_mus, and custom_stellar_model.
    ld_data_path : string
        Path to exotic-ld-data directory. As of version>=3.2.0 this path
        specifies where stellar and instrument data are automatically
        downloaded and stored. Only the required data is downloaded, and
        if the data has previsouly been used, then no download is required.
        The directory will be automatically created on the first call.
        It remains an option, and is backwards compatible, to download
        all the data from zenodo and specify the path.
    interpolate_type : string
        Choose between 'nearest' and 'trilinear'.
    custom_wavelengths : numpy.ndarray, shape (n,)
        If ld_model='custom', pass the wavelengths of you stellar model
        in angstroms.
    custom_mus : numpy.ndarray, shape (m,)
        If ld_model='custom', pass the mu values at which you stellar
        model is defined.
    custom_stellar_model : numpy.ndarray, shape (n, m)
        If ld_model='custom', pass the specific intensity of your stellar
        for each wavelength and mu value. Note specific intensity must
        be in units of [n_photons / s / cm^2 / Angstrom / steradian].
    ld_data_version : string
        Version number of the data files. Implemented at 3.2.0, and
        this corresponds to files with no version number appended.
        Recommend not changing this from the default value.
    verbose : int
        Level of printed information during calculation. Default: 1.
        0 (no info), 1 (warnings/downloads), 2 (step-by-step info).

    Examples
    --------
    >>> from exotic_ld import StellarLimbDarkening
    >>> sld = StellarLimbDarkening(
            M_H=0.1, Teff=6045, logg=4.2, ld_model='mps1',
            ld_data_path='path/to/ExoTiC-LD_data')
    >>> c1, c2 = sld.compute_quadratic_ld_coeffs(
            wavelength_range=np.array([20000., 30000.]),
            mode='JWST_NIRSpec_Prism')

    """

    def __init__(self, M_H=None, Teff=None, logg=None, ld_model="mps1",
                 ld_data_path="exotic_ld_data", interpolate_type="nearest",
                 custom_wavelengths=None, custom_mus=None,
                 custom_stellar_model=None, ld_data_version="3.2.0",
                 verbose=1):
        self.verbose = verbose

        # Stellar input parameters.
        self.M_H_input = float(M_H) if M_H is not None else None
        self.Teff_input = int(Teff) if Teff is not None else None
        self.logg_input = float(logg) if logg is not None else None
        self.interpolate_type = interpolate_type
        if self.verbose > 1:
            print("Input stellar parameters are M_H={}, Teff={}, logg={}."
                  .format(self.M_H_input, self.Teff_input, self.logg_input))

        # Set stellar grid.
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

        # Load/build stellar model.
        self.stellar_wavelengths = None
        self.mus = None
        self.stellar_intensities = None
        self.I_mu = None
        if self.ld_model == "custom":
            self.stellar_wavelengths = custom_wavelengths
            self.mus = custom_mus
            self.stellar_intensities = custom_stellar_model
            if self.verbose > 1:
                print("Using custom stellar model.")
        else:
            self._load_stellar_model()
        self._check_stellar_model()

    def __repr__(self):
        return "Stellar limb darkening: {} models.".format(self.ld_model)

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
            Alternatively, use 'custom' mode. In this case the custom
            wavelength and custom throughput must also be specified.
        custom_wavelengths : array_like, optional
            Wavelengths corresponding to custom_throughput [angstroms].
        custom_throughput : array_like, optional
            Throughputs corresponding to custom_wavelengths.
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
            Alternatively, use 'custom' mode. In this case the custom
            wavelength and custom throughput must also be specified.
        custom_wavelengths : array_like, optional
            Wavelengths corresponding to custom_throughput [angstroms].
        custom_throughput : array_like, optional
            Throughputs corresponding to custom_wavelengths.
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
        # Compute I(mu) for a given response function.
        self._integrate_I_mu(wavelength_range, mode,
                             custom_wavelengths, custom_throughput)

        # Fit limb-darkening law.
        return self._fit_ld_law(quadratic_ld_law, mu_min, return_sigmas)

    def compute_kipping_ld_coeffs(self, wavelength_range, mode,
                                  custom_wavelengths=None,
                                  custom_throughput=None,
                                  mu_min=0.10, return_sigmas=False):
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
        wavelength_range : array_like, (start, end)
            Wavelength range over which to compute the limb-darkening
            coefficients. Wavelengths must be given in angstroms and
            the values must fall within the supported range of the
            corresponding instrument mode.
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
            Alternatively, use 'custom' mode. In this case the custom
            wavelength and custom throughput must also be specified.
        custom_wavelengths : array_like, optional
            Wavelengths corresponding to custom_throughput [angstroms].
        custom_throughput : array_like, optional
            Throughputs corresponding to custom_wavelengths.
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
        # Compute I(mu) for a given response function.
        self._integrate_I_mu(wavelength_range, mode,
                             custom_wavelengths, custom_throughput)

        # Fit limb-darkening law.
        return self._fit_ld_law(kipping_ld_law, mu_min, return_sigmas)

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
            Alternatively, use 'custom' mode. In this case the custom
            wavelength and custom throughput must also be specified.
        custom_wavelengths : array_like, optional
            Wavelengths corresponding to custom_throughput [angstroms].
        custom_throughput : array_like, optional
            Throughputs corresponding to custom_wavelengths.
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
            Alternatively, use 'custom' mode. In this case the custom
            wavelength and custom throughput must also be specified.
        custom_wavelengths : array_like, optional
            Wavelengths corresponding to custom_throughput [angstroms].
        custom_throughput : array_like, optional
            Throughputs corresponding to custom_wavelengths.
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
            Alternatively, use 'custom' mode. In this case the custom
            wavelength and custom throughput must also be specified.
        custom_wavelengths : array_like, optional
            Wavelengths corresponding to custom_throughput [angstroms].
        custom_throughput : array_like, optional
            Throughputs corresponding to custom_wavelengths.
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
        # Compute I(mu) for a given response function.
        self._integrate_I_mu(wavelength_range, mode,
                             custom_wavelengths, custom_throughput)

        # Fit limb-darkening law.
        return self._fit_ld_law(nonlinear_4param_ld_law, mu_min, return_sigmas)

    def _load_stellar_model(self):
        if self.verbose > 1:
            print("Loading stellar model from {} grid.".format(self.ld_model))

        sg = StellarGrids(self.M_H_input, self.Teff_input,
                          self.logg_input, self.ld_model,
                          self.ld_data_path, self.remote_ld_data_path,
                          self.ld_data_version, self.interpolate_type,
                          self.verbose)
        self.stellar_wavelengths, self.mus, self.stellar_intensities = \
            sg.get_stellar_data()

        if self.verbose > 1:
            print("Stellar model loaded.")

    def _check_stellar_model(self):
        if not self.stellar_intensities.ndim == 2:
            raise ValueError("Stellar intensities must be 2D, with shape "
                             "(n wavelengths, n mu values)")
        if not self.stellar_wavelengths.shape[0] == \
               self.stellar_intensities.shape[0]:
            raise ValueError("Stellar wavelengths must have the same shape as "
                             "stellar_intensities.shape[0].")
        if not self.mus.shape[0] == self.stellar_intensities.shape[1]:
            raise ValueError("mu values must have the same shape as "
                             "stellar_intensities.shape[1].")

    def _read_sensitivity_data(self, mode):
        local_sensitivity_file_path = os.path.join(
            self.ld_data_path,
            "Sensitivity_files/{}_throughput{}.csv".format(mode, self.ld_data_version))
        remote_sensitivity_file_path = os.path.join(
            self.remote_ld_data_path,
            "Sensitivity_files/{}_throughput{}.csv".format(mode, self.ld_data_version))

        # Check if exists locally.
        if not os.path.exists(local_sensitivity_file_path):
            download(remote_sensitivity_file_path,
                     local_sensitivity_file_path, self.verbose)
            if self.verbose > 1:
                print("Downloaded {}.".format(local_sensitivity_file_path))

        sensitivity_data = np.loadtxt(local_sensitivity_file_path,
                                      skiprows=1, delimiter=",")
        sensitivity_wavelengths = sensitivity_data[:, 0]
        sensitivity_throughputs = sensitivity_data[:, 1]

        return sensitivity_wavelengths, sensitivity_throughputs

    def _integrate_I_mu(self, wavelength_range, mode, custom_wavelengths,
                        custom_throughput):
        if mode == "custom":
            # Custom throughput provided.
            s_wavelengths = custom_wavelengths
            s_throughputs = custom_throughput
            if self.verbose > 1:
                print("Using custom instrument throughput with wavelength "
                      "range {}-{} A.".format(s_wavelengths[0],
                                              s_wavelengths[-1]))
        else:
            # Read in mode specific throughput.
            s_wavelengths, s_throughputs = self._read_sensitivity_data(mode)
            if self.verbose > 1:
                print("Loading instrument mode={} with wavelength range "
                      "{}-{} A.".format(mode, s_wavelengths[0],
                                        s_wavelengths[-1]))

        # Select wavelength range.
        wavelength_range = np.sort(np.array(wavelength_range))
        s_mask = np.logical_and(wavelength_range[0] < s_wavelengths,
                                s_wavelengths < wavelength_range[1])
        if wavelength_range[1] < s_wavelengths[0] \
                or s_wavelengths[-1] < wavelength_range[0]:
            raise ValueError(
                "Wavelength range {}-{} A has no overlap with instrument "
                "mode {}'s range {}-{} A.".format(
                    wavelength_range[0], wavelength_range[1], mode,
                    s_wavelengths[0], s_wavelengths[-1]))

        i_mask = np.logical_and(wavelength_range[0] < self.stellar_wavelengths,
                                self.stellar_wavelengths < wavelength_range[1])
        if wavelength_range[1] < self.stellar_wavelengths[0] \
                or self.stellar_wavelengths[-1] < wavelength_range[0]:
            raise ValueError(
                "Wavelength range {}-{} A has no overlap with stellar "
                "spectra's range {}-{} A.".format(
                    wavelength_range[0], wavelength_range[1],
                    self.stellar_wavelengths[0], self.stellar_wavelengths[-1]))

        s_wvs = s_wavelengths[s_mask]
        s_thp = s_throughputs[s_mask]
        i_wvs = self.stellar_wavelengths[i_mask]
        i_int = self.stellar_intensities[i_mask]

        # Ready sensitivity interpolator.
        if s_wvs.shape[0] >= 2:
            s_interp_func = interp1d(s_wvs, s_thp, kind="linear",
                                     bounds_error=False, fill_value=0.)
        else:
            mean_wv = np.mean(wavelength_range)
            match_wv_idx = np.argmin(np.abs(s_wavelengths - mean_wv))
            match_s = s_throughputs[match_wv_idx]
            s_interp_func = lambda _sw: np.ones(_sw.shape) * match_s

        # Pre-compute Gauss-legendre roots and rescale to lims.
        roots, weights = roots_legendre(500)
        a = wavelength_range[0]
        b = wavelength_range[1]
        t = (b - a) / 2 * roots + (a + b) / 2

        if self.verbose > 1:
            print("Integrating I(mu) for wavelength limits of {}-{} A."
                  .format(a, b))

        # Iterate mu values computing intensity.
        self.I_mu = np.zeros(i_int.shape[1])
        for mu_idx in range(self.mus.shape[0]):

            # Ready intensity interpolator.
            if i_wvs.shape[0] >= 2:
                i_interp_func = interp1d(
                    i_wvs, i_int[:, mu_idx], kind="linear",
                    bounds_error=False, fill_value=0.)
            else:
                mean_wv = np.mean(wavelength_range)
                match_wv_idx = np.argmin(np.abs(self.stellar_wavelengths - mean_wv))
                match_i = self.stellar_intensities[match_wv_idx, mu_idx]
                i_interp_func = lambda _iw: np.ones(_iw.shape) * match_i

            def integrand(_lambda):
                return s_interp_func(_lambda) * i_interp_func(_lambda)

            # Approximate integral.
            self.I_mu[mu_idx] = (b - a) / 2. * integrand(t).dot(weights)

        # Set I(mu=1) = 1.
        if not self.I_mu[0] == 0.:
            self.I_mu /= self.I_mu[0]
        else:
            raise ValueError("Zero intensity in this passband, check your "
                             "wavelength range is correct and in angstroms.")

        if self.verbose > 1:
            print("Integral done for I(mu).")

    def _fit_ld_law(self, ld_law_func, mu_min, return_sigmas):
        # Truncate mu range to be fitted.
        mu_mask = self.mus >= mu_min

        if np.sum(mu_mask) < 2:
            raise ValueError("mu_min={} set too high, must be >= 2 mu "
                             "values remaining.".format(mu_min))

        if not ld_law_func == kipping_ld_law:
            if self.verbose > 1:
                print("Fitting limb-darkening law to {} I(mu) data points "
                      "where {} <= mu <= 1, with the Levenberg-Marquardt "
                      "algorithm.".format(np.sum(mu_mask), mu_min))

            # Fit limb-darkening law: Levenberg-Marquardt (LM), guess=1.
            popt, pcov = curve_fit(ld_law_func,
                                   self.mus[mu_mask],
                                   self.I_mu[mu_mask],
                                   method="lm")

        else:
            if self.verbose > 1:
                print("Fitting limb-darkening law to {} I(mu) data points "
                      "where {} <= mu <= 1, with the constrained Trust-Region "
                      "Reflective algorithm.".format(np.sum(mu_mask), mu_min))

            # Fit limb-darkening law: Trust-Region Reflective (TRF), guess=0.5.
            # For the Kipping law, constrain q1, q2 to [0, 1].
            popt, pcov = curve_fit(kipping_ld_law,
                                   self.mus[mu_mask],
                                   self.I_mu[mu_mask],
                                   p0=(0.5, 0.5),
                                   bounds=((0, 0), (1, 1)),
                                   method="trf")

        if self.verbose > 1:
            print("Fit done, resulting coefficients are {}.".format(popt))

        if return_sigmas:
            return tuple(popt), tuple(np.sqrt(np.diag(pcov)))
        else:
            return tuple(popt)
