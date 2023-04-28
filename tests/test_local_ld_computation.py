import unittest
import numpy as np

from exotic_ld import StellarLimbDarkening
from exotic_ld.ld_laws import linear_ld_law, quadratic_ld_law, \
    squareroot_ld_law, nonlinear_3param_ld_law, nonlinear_4param_ld_law


class TestLDC(unittest.TestCase):
    """ Test limb-darkening computation locally, with data downloaded. """

    def __init__(self, *args, **kwargs):
        super(TestLDC, self).__init__(*args, **kwargs)

        # Make reproducible.
        np.random.seed(123)

        # Define local test paths.
        self.local_data_path = '../../../data/exotic_ld_data'

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

    def _run_all_ld_laws(self, sld_object):
        test_mu = np.linspace(1., 0., 10)
        for wr, im in zip(self.wave_ranges, self.instrument_modes):

            # Linear law.
            u1 = sld_object.compute_linear_ld_coeffs(
                wavelength_range=wr, mode=im)
            I_mu = linear_ld_law(test_mu, u1)
            self.assertEqual(I_mu[0], 1.)
            self.assertFalse(np.any(np.diff(I_mu) > 0.))

            # Quadratic law.
            u1, u2 = sld_object.compute_quadratic_ld_coeffs(
                wavelength_range=wr, mode=im)
            I_mu = quadratic_ld_law(test_mu, u1, u2)
            self.assertEqual(I_mu[0], 1.)
            self.assertFalse(np.any(np.diff(I_mu) > 0.))

            # Square-root law.
            u1, u2 = sld_object.compute_squareroot_ld_coeffs(
                wavelength_range=wr, mode=im)
            I_mu = squareroot_ld_law(test_mu, u1, u2)
            self.assertEqual(I_mu[0], 1.)
            self.assertFalse(np.any(np.diff(I_mu) > 0.))

            # 3-param non-linear law.
            u1, u2, u3 = sld_object.compute_3_parameter_non_linear_ld_coeffs(
                wavelength_range=wr, mode=im)
            I_mu = nonlinear_3param_ld_law(test_mu, u1, u2, u3)
            self.assertEqual(I_mu[0], 1.)
            self.assertFalse(np.any(np.diff(I_mu) > 0.))

            # 4-param non-linear law.
            u1, u2, u3, u4 = sld_object.compute_4_parameter_non_linear_ld_coeffs(
                wavelength_range=wr, mode=im)
            I_mu = nonlinear_4param_ld_law(test_mu, u1, u2, u3, u4)
            self.assertEqual(I_mu[0], 1.)
            self.assertFalse(np.any(np.diff(I_mu) > 0.))

    def test_ld_computation_1D_grid(self):
        """ Test ld computation, defaults to Kurucz grid. """
        sld = StellarLimbDarkening(
            M_H=0.0, Teff=4500, logg=4.5, ld_model='1D',
            ld_data_path=self.local_data_path)
        self._run_all_ld_laws(sld)

    def test_ld_computation_3D_grid(self):
        """ Test ld computation, defaults to Stagger grid. """
        sld = StellarLimbDarkening(
            M_H=0.0, Teff=4500, logg=4.5, ld_model='3D',
            ld_data_path=self.local_data_path)
        self._run_all_ld_laws(sld)

    def test_ld_computation_kurucz_grid(self):
        """ Test ld computation, Kurucz grid. """
        sld = StellarLimbDarkening(
            M_H=0.0, Teff=4500, logg=4.5, ld_model='kurucz',
            ld_data_path=self.local_data_path)
        self._run_all_ld_laws(sld)

    def test_ld_computation_stagger_grid(self):
        """ Test ld computation, Stagger grid. """
        sld = StellarLimbDarkening(
            M_H=0.0, Teff=4500, logg=4.5, ld_model='stagger',
            ld_data_path=self.local_data_path)
        self._run_all_ld_laws(sld)

    def test_ld_computation_mps1_grid(self):
        """ Test ld computation, MPS-ATLAS-1 grid. """
        sld = StellarLimbDarkening(
            M_H=0.0, Teff=4500, logg=4.5, ld_model='mps1',
            ld_data_path=self.local_data_path)
        self._run_all_ld_laws(sld)

    def test_ld_computation_mps2_grid(self):
        """ Test ld computation, MPS-ATLAS-2 grid. """
        sld = StellarLimbDarkening(
            M_H=0.0, Teff=4500, logg=4.5, ld_model='mps2',
            ld_data_path=self.local_data_path)
        self._run_all_ld_laws(sld)

    def test_min_mu_setting(self):
        """ Test variable min mu for ld fits.. """
        sld = StellarLimbDarkening(
            M_H=0.0, Teff=4500, logg=4.5, ld_model='mps1',
            ld_data_path=self.local_data_path)

        u = sld.compute_quadratic_ld_coeffs(
            wavelength_range=[10000, 20000], mode='JWST_NIRSpec_Prism',
            mu_min=0.0)
        self.assertEqual(type(u), tuple)
        self.assertEqual(len(u), 2)

        u = sld.compute_quadratic_ld_coeffs(
            wavelength_range=[10000, 20000], mode='JWST_NIRSpec_Prism',
            mu_min=0.10)
        self.assertEqual(type(u), tuple)
        self.assertEqual(len(u), 2)

        u = sld.compute_quadratic_ld_coeffs(
            wavelength_range=[10000, 20000], mode='JWST_NIRSpec_Prism',
            mu_min=0.20)
        self.assertEqual(type(u), tuple)
        self.assertEqual(len(u), 2)

    def test_ld_probabilistic(self):
        """ Test ld computation in probabilistic mode. """
        sld = StellarLimbDarkening(
            M_H=0.0, Teff=4500, logg=4.5, ld_model='mps1',
            ld_data_path=self.local_data_path)
        u, u_sigma = sld.compute_4_parameter_non_linear_ld_coeffs(
            wavelength_range=[10000, 20000], mode='JWST_NIRSpec_Prism',
            return_sigmas=True)

        self.assertEqual(type(u), tuple)
        self.assertEqual(len(u), 4)
        self.assertEqual(type(u_sigma), tuple)
        self.assertEqual(len(u_sigma), 4)

    def test_ld_custom_throughput(self):
        """ Test ld computation w/ custom throughput. """
        sld = StellarLimbDarkening(
            M_H=0.0, Teff=4500, logg=4.5, ld_model='mps1',
            ld_data_path=self.local_data_path)
        u = sld.compute_quadratic_ld_coeffs(
            wavelength_range=[10000, 20000], mode='custom',
            custom_wavelengths=np.linspace(10000, 20000, 100),
            custom_throughput=np.ones(100))

        self.assertEqual(type(u), tuple)
        self.assertEqual(len(u), 2)


if __name__ == '__main__':
    unittest.main()
