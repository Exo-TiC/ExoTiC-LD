import unittest
import numpy as np

from exotic_ld import StellarLimbDarkening


class TestLDC(unittest.TestCase):
    """ Test limb-darkening computation. """

    def __init__(self, *args, **kwargs):
        super(TestLDC, self).__init__(*args, **kwargs)

        # Make reproducible.
        np.random.seed(3)

        # Define supported modes.
        self.instrument_modes = [
            'HST_STIS_G430L', 'HST_STIS_G750L', 'HST_WFC3_G280p1',
            'HST_WFC3_G280n1', 'HST_WFC3_G102', 'HST_WFC3_G141',
            'JWST_NIRSpec_Prism', 'JWST_NIRSpec_G395H',
            'JWST_NIRSpec_G395M', 'JWST_NIRSpec_G235H',
            'JWST_NIRSpec_G235M', 'JWST_NIRSpec_G140H-f100',
            'JWST_NIRSpec_G140M-f100', 'JWST_NIRSpec_G140H-f070',
            'JWST_NIRSpec_G140M-f070', 'JWST_NIRISS_SOSSo1',
            'JWST_NIRISS_SOSSo2', 'JWST_NIRCam_F322W2',
            'JWST_NIRCam_F444', 'JWST_MIRI_LRS',
            'Spitzer_IRAC_Ch1', 'Spitzer_IRAC_Ch2', 'TESS']
        self.wave_ranges = [[2882., 5740.], [5238., 10332.],
                            [1900., 10000.],
                            [1900., 10000.], [7450., 12200.],
                            [9700.66, 17906.90],
                            [6000., 53000.], [28700., 51776.6],
                            [28700., 51787.1], [16600., 30703.3],
                            [16600., 31204.4], [9700., 18309.7],
                            [9700., 18694.4], [8243.47, 12700.],
                            [7000., 12700.], [8300., 28100.],
                            [6300., 12600.], [24000., 42199.],
                            [24800., 49984.25], [50000., 138632.61],
                            [29650., 41650.], [37040., 52640.],
                            [5670., 11270.]]

    def test_stellar_models_available_1d(self):
        """ Test 1d stellar models available. """
        M_Hs = np.linspace(-6., 2., 10)
        Teffs = np.linspace(3000., 7000., 10)
        loggs = np.linspace(3., 6., 10)
        for M_H in M_Hs:
            for Teff in Teffs:
                for logg in loggs:
                    StellarLimbDarkening(
                        M_H, Teff, logg, ld_model='1D',
                        ld_data_path='../data')

    def test_stellar_models_available_3d(self):
        """ Test 3d stellar models available. """
        M_Hs = np.linspace(-3., 1., 10)
        Teffs = np.linspace(3000., 8000., 10)
        loggs = np.linspace(1., 6., 10)
        for M_H in M_Hs:
            for Teff in Teffs:
                for logg in loggs:
                    StellarLimbDarkening(
                        M_H, Teff, logg, ld_model='3D',
                        ld_data_path='../data')

    def test_instrument_modes_1d(self):
        """ Test instrument modes, 1d stellar models. """
        sld = StellarLimbDarkening(
            M_H=0.1, Teff=6000, logg=3.0, ld_model='1D',
            ld_data_path='../data')
        for im, wr in zip(self.instrument_modes, self.wave_ranges):
            sld.compute_linear_ld_coeffs(
                wavelength_range=wr, mode=im)
            sld.compute_quadratic_ld_coeffs(
                wavelength_range=wr, mode=im)
            sld.compute_3_parameter_non_linear_ld_coeffs(
                wavelength_range=wr, mode=im)
            sld.compute_4_parameter_non_linear_ld_coeffs(
                wavelength_range=wr, mode=im)

    def test_instrument_modes_3d(self):
        """ Test instrument modes, 3d stellar models. """
        sld = StellarLimbDarkening(
            M_H=0.1, Teff=6000, logg=3.0, ld_model='3D',
            ld_data_path='../data')
        for im, wr in zip(self.instrument_modes, self.wave_ranges):
            sld.compute_linear_ld_coeffs(
                wavelength_range=wr, mode=im)
            sld.compute_quadratic_ld_coeffs(
                wavelength_range=wr, mode=im)
            sld.compute_3_parameter_non_linear_ld_coeffs(
                wavelength_range=wr, mode=im)
            sld.compute_4_parameter_non_linear_ld_coeffs(
                wavelength_range=wr, mode=im)

    def test_custom_mode(self):
        """ Test custom instrument mode. """
        # Create custom throughput.
        custom_wavelengths = np.linspace(10000, 20000, 100)
        custom_throughput = np.exp(
            -0.5 * ((custom_wavelengths - 15000.) / 5000.)**2)

        sld = StellarLimbDarkening(
            M_H=0.1, Teff=6000, logg=3.0, ld_model='3D',
            ld_data_path='../data')
        sld.compute_4_parameter_non_linear_ld_coeffs(
            wavelength_range=[13000, 17000], mode='custom',
            custom_wavelengths=custom_wavelengths,
            custom_throughput=custom_throughput)


if __name__ == '__main__':
    unittest.main()
