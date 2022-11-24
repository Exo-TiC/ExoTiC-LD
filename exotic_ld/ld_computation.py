import os
import numpy as np
import pandas as pd
from astropy.modeling.fitting import LevMarLSQFitter
from scipy.interpolate import interp1d, splev, splrep

from exotic_ld.ld_grids import StellarGrids
from exotic_ld.ld_laws import quadratic_limb_darkening, \
    nonlinear_limb_darkening


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
        print("Input stellar parameters are M_H={}, Teff={}, logg={}.".format(
            self.M_H_input, self.Teff_input, self.logg_input))
        self.interpolate_type = interpolate_type

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
        if self.ld_model == "custom":
            print("Using custom stellar model.")
            # Todo: validate wvs, mus, intensities in correct shape/order.
            self.stellar_wavelengths = custom_wavelengths
            self.mus = custom_mus
            self.stellar_intensities = custom_stellar_model
        else:
            self._load_stellar_model()

    def __repr__(self):
        return 'Stellar limb darkening: {} models.'.format(self.ld_model)

    def _load_stellar_model(self, ):
        """ Load stellar model. """
        print("Loading stellar model from {} grid.".format(self.ld_model))
        sg = StellarGrids(self.M_H_input, self.Teff_input, self.logg_input,
                          self.ld_model, self.ld_data_path, self.interpolate_type,
                          self.verbose)
        self.stellar_wavelengths, self.mus, self.stellar_intensities = \
            sg.get_stellar_data()
        print("Stellar model loaded.")





    def compute_linear_ld_coeffs(self, wavelength_range, mode,
                                 custom_wavelengths=None,
                                 custom_throughput=None):
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
        # Compute the stellar limb-darkening.
        mu, intensity = self._limb_dark_fit(wavelength_range, mode,
                                            custom_wavelengths,
                                            custom_throughput)

        # Fit linear limb-darkening law.
        fitter = LevMarLSQFitter()
        linear = nonlinear_limb_darkening()
        linear.c0.fixed = True
        linear.c2.fixed = True
        linear.c3.fixed = True
        linear = fitter(linear, mu, intensity)

        return (linear.c1.value, )

    def compute_quadratic_ld_coeffs(self, wavelength_range, mode,
                                    custom_wavelengths=None,
                                    custom_throughput=None):
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
        # Compute the stellar limb-darkening.
        mu, intensity = self._limb_dark_fit(wavelength_range, mode,
                                            custom_wavelengths,
                                            custom_throughput)

        # Fit linear limb-darkening law.
        fitter = LevMarLSQFitter()
        quadratic = quadratic_limb_darkening()
        quadratic = fitter(quadratic, mu, intensity)

        return quadratic.parameters

    def compute_3_parameter_non_linear_ld_coeffs(self, wavelength_range, mode,
                                                 custom_wavelengths=None,
                                                 custom_throughput=None):
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
        # Compute the stellar limb-darkening.
        mu, intensity = self._limb_dark_fit(wavelength_range, mode,
                                            custom_wavelengths,
                                            custom_throughput)

        # Fit linear limb-darkening law.
        fitter = LevMarLSQFitter()
        corot_3_param = nonlinear_limb_darkening()
        corot_3_param.c0.fixed = True
        corot_3_param = fitter(corot_3_param, mu, intensity)

        return corot_3_param.parameters[1:]

    def compute_4_parameter_non_linear_ld_coeffs(self, wavelength_range, mode,
                                                 custom_wavelengths=None,
                                                 custom_throughput=None):
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
        # Compute the stellar limb-darkening.
        mu, intensity = self._limb_dark_fit(wavelength_range, mode,
                                            custom_wavelengths,
                                            custom_throughput)

        # Fit linear limb-darkening law.
        fitter = LevMarLSQFitter()
        corot_4_param = nonlinear_limb_darkening()
        corot_4_param = fitter(corot_4_param, mu, intensity)

        return corot_4_param.parameters

    def _limb_dark_fit(self, wavelength_range, mode, custom_wavelengths,
                       custom_throughput):
        """ Compute stellar limb-darkening coefficients. """
        if mode == 'custom':
            # Custom throughput provided.
            sen_wavelengths = custom_wavelengths
            sen_throughputs = custom_throughput
        else:
            # Read in mode specific throughput.
            sen_wavelengths, sen_throughputs = \
                self._read_throughput_data(mode)

        # Pad arrays.
        sen_wavelengths = self._pad_array(
            sen_wavelengths,
            [sen_wavelengths[0] - 2., sen_wavelengths[0] - 1.],
            [sen_wavelengths[-1] + 1., sen_wavelengths[-1] + 2.])
        sen_throughputs = self._pad_array(
            sen_throughputs, [0., 0.], [0., 0.])
        bin_wavelengths = self._pad_array(
            wavelength_range,
            [wavelength_range[0] - 2., wavelength_range[0] - 1.],
            [wavelength_range[-1] + 1., wavelength_range[-1] + 2.])

        # Interpolate throughput onto stellar model wavelengths.
        interpolator = interp1d(sen_wavelengths, sen_throughputs,
                                bounds_error=False, fill_value=0)
        sen_interp = interpolator(self.stellar_wavelengths)

        # Interpolate bin mask onto stellar model wavelengths.
        # todo: imporve, why bin mask at all? just slect only wvs required.
        bin_mask = np.zeros(bin_wavelengths.shape[0])
        bin_mask[2:-2] = 1.
        interpolator = interp1d(bin_wavelengths, bin_mask,
                                bounds_error=False, fill_value=0)
        bin_mask_interp = interpolator(self.stellar_wavelengths)
        if np.all(bin_mask_interp == 0):
            # High resolution, mask interpolated to nothing.
            # Select nearest point in stellar wavelength grid.
            mid_bin_wavelengths = np.mean(bin_wavelengths)
            nearest_stellar_wavelength_idx = (
                abs(mid_bin_wavelengths - self.stellar_wavelengths)).argmin()
            bin_mask_interp[nearest_stellar_wavelength_idx] = 1.

        # Integrate per mu over spectra computing synthetic photometric points.
        phot = np.zeros(self.stellar_intensities.shape[0])
        f = self.stellar_wavelengths * sen_interp * bin_mask_interp
        tot = self._int_tabulated(self.stellar_wavelengths, f)
        if tot == 0.:
            raise ValueError(
                'Input wavelength range {}-{} does not overlap with instrument '
                'mode {} with range {}-{}.'.format(
                    wavelength_range[0], wavelength_range[-1], mode,
                    sen_wavelengths[0], sen_wavelengths[-1]))

        for i in range(self.mus.shape[0]):
            f_cal = self.stellar_intensities[i, :]
            phot[i] = self._int_tabulated(
                self.stellar_wavelengths, f * f_cal, sort=True) / tot
        if self.ld_model == '1D' or self.ld_model == 'kurucz':
            yall = phot / phot[0]
        elif self.ld_model == '3D':
            yall = phot / phot[10]

        return self.mus[1:], yall[1:]

    def _read_throughput_data(self, mode):
        """ Read in throughput data. """
        sensitivity_file = os.path.join(
            self.ld_data_path,
            'Sensitivity_files/{}_throughput.csv'.format(mode))
        sensitivity_data = pd.read_csv(sensitivity_file)
        sensitivity_wavelengths = sensitivity_data['wave'].values
        sensitivity_throughputs = sensitivity_data['tp'].values

        return sensitivity_wavelengths, sensitivity_throughputs

    def _pad_array(self, array, values_start, values_end):
        """ Pad array with values. """
        array = np.concatenate(
            (np.array(values_start), array, np.array(values_end)))
        return array

    def _int_tabulated(self, X, F, sort=False):
        Xsegments = len(X) - 1

        # Sort vectors into ascending order.
        if not sort:
            ii = np.argsort(X)
            X = X[ii]
            F = F[ii]

        while (Xsegments % 4) != 0:
            Xsegments = Xsegments + 1

        Xmin = np.min(X)
        Xmax = np.max(X)

        # Uniform step size.
        h = (Xmax + 0.0 - Xmin) / Xsegments
        # Compute the interpolates at Xgrid.
        # x values of interpolates >> Xgrid = h * FINDGEN(Xsegments + 1L)+Xmin
        z = splev(h * np.arange(Xsegments + 1) + Xmin, splrep(X, F))

        # Compute the integral using the 5-point Newton-Cotes formula.
        ii = (np.arange((len(z) - 1) / 4, dtype=int) + 1) * 4

        return np.sum(2.0 * h * (7.0 * (z[ii - 4] + z[ii])
                                 + 32.0 * (z[ii - 3] + z[ii - 1])
                                 + 12.0 * z[ii - 2]) / 45.0)
