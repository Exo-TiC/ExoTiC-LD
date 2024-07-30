import os
import shutil
import unittest
import numpy as np

from exotic_ld import StellarLimbDarkening
from exotic_ld.ld_laws import *


class TestLDC(unittest.TestCase):
    """ Test limb-darkening computation locally, with data downloaded. """

    @classmethod
    def setUpClass(cls):
        # Make reproducible.
        np.random.seed(123)

        # Define local test paths.
        cls.local_data_path = 'test_exotic_ld_data'

        # Clear cached data.
        if os.path.exists(cls.local_data_path):
            shutil.rmtree(cls.local_data_path)

        # Define supported modes.
        cls.instrument_modes = [
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
        cls.wave_ranges = [[2882., 5740.], [5238., 10332.],
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

        # Constants.
        cls.h = 6.62607004e-34  # Planck's constant [SI].
        cls.c = 2.99792458e8  # Speed of light [SI].
        cls.k_b = 1.38064852e-23  # Boltzmann constant [SI].

    @classmethod
    def tearDownClass(cls):
        # Tidy up.
        if os.path.exists(cls.local_data_path):
            shutil.rmtree(cls.local_data_path)

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

    def test_ld_computation_phoenix_grid(self):
        """ Test ld computation, Phoenix grid. """
        sld = StellarLimbDarkening(
            M_H=-1.5, Teff=4500, logg=4.5, ld_model='phoenix',
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

    def test_ld_computation_custom_stellar_and_throughput_model(self):
        """ Test ld computation, custom stellar and throughput model. """
        s_wvs, mus, stellar_intensity = \
            self._generate_synthetic_stellar_models()
        sld = StellarLimbDarkening(
            ld_model="custom",
            custom_wavelengths=s_wvs,
            custom_mus=mus,
            custom_stellar_model=stellar_intensity)
        self._run_all_custom_ld_laws(sld)

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

            # Kipping law.
            q1, q2 = sld_object.compute_kipping_ld_coeffs(
                wavelength_range=wr, mode=im)
            I_mu = kipping_ld_law(test_mu, q1, q2)
            self.assertEqual(I_mu[0], 1.)
            self.assertFalse(np.any(np.diff(I_mu) > 0.))

            # power2 law.
            q1, q2 = sld_object.compute_power2_ld_coeffs(
                wavelength_range=wr, mode=im)
            I_mu = power2_ld_law(test_mu, q1, q2)
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

    def _run_all_custom_ld_laws(self, sld_object):
        test_mu = np.linspace(1., 0., 10)
        for s_wr in self.wave_ranges:

            # Generate custom throughput.
            s_wvs, throughput = self._generate_synthetic_throughput(s_wr)

            # Select random interval within throughput range.
            wr = np.sort(np.random.uniform(s_wr[0], s_wr[1], size=2))

            # Linear law.
            u1 = sld_object.compute_linear_ld_coeffs(
                wavelength_range=wr, mode="custom",
                custom_wavelengths=s_wvs,
                custom_throughput=throughput)
            I_mu = linear_ld_law(test_mu, u1)
            self.assertEqual(I_mu[0], 1.)
            self.assertFalse(np.any(np.diff(I_mu) > 0.))

            # Quadratic law.
            u1, u2 = sld_object.compute_quadratic_ld_coeffs(
                wavelength_range=wr, mode="custom",
                custom_wavelengths=s_wvs,
                custom_throughput=throughput)
            I_mu = quadratic_ld_law(test_mu, u1, u2)
            self.assertEqual(I_mu[0], 1.)
            self.assertFalse(np.any(np.diff(I_mu) > 0.))

            # Kipping law.
            q1, q2 = sld_object.compute_kipping_ld_coeffs(
                wavelength_range=wr, mode="custom",
                custom_wavelengths=s_wvs,
                custom_throughput=throughput)
            I_mu = kipping_ld_law(test_mu, q1, q2)
            self.assertEqual(I_mu[0], 1.)
            self.assertFalse(np.any(np.diff(I_mu) > 0.))

            # Square-root law.
            u1, u2 = sld_object.compute_squareroot_ld_coeffs(
                wavelength_range=wr, mode="custom",
                custom_wavelengths=s_wvs,
                custom_throughput=throughput)
            I_mu = squareroot_ld_law(test_mu, u1, u2)
            self.assertEqual(I_mu[0], 1.)
            # self.assertFalse(np.any(np.diff(I_mu) > 0.))

            # 3-param non-linear law.
            u1, u2, u3 = sld_object.compute_3_parameter_non_linear_ld_coeffs(
                wavelength_range=wr, mode="custom",
                custom_wavelengths=s_wvs,
                custom_throughput=throughput)
            I_mu = nonlinear_3param_ld_law(test_mu, u1, u2, u3)
            self.assertEqual(I_mu[0], 1.)
            self.assertFalse(np.any(np.diff(I_mu) > 0.))

            # 4-param non-linear law.
            u1, u2, u3, u4 = sld_object.compute_4_parameter_non_linear_ld_coeffs(
                wavelength_range=wr, mode="custom",
                custom_wavelengths=s_wvs,
                custom_throughput=throughput)
            I_mu = nonlinear_4param_ld_law(test_mu, u1, u2, u3, u4)
            self.assertEqual(I_mu[0], 1.)
            self.assertFalse(np.any(np.diff(I_mu) > 0.))

    def _generate_synthetic_stellar_models(self):
        # Generate I(lambda, mu).
        wvs = np.linspace(0.01e-6, 50e-6, 1000)
        mus = np.linspace(1., 0.01, 10)
        temps = np.linspace(5000., 4500., 10)
        stellar_intensity = []
        for mu, temp in zip(mus, temps):
            stellar_intensity.append(self._plancks_law(wvs, temp))
        return wvs * 1.e10, mus, np.array(stellar_intensity).T

    def _plancks_law(self, wav, temp):
        a = 2.0 * self.h * self.c**2
        b = self.h * self.c / (wav * self.k_b * temp)
        intensity = a / (wav**5 * (np.exp(b) - 1.0))
        return intensity

    def _generate_synthetic_throughput(self, wr):
        wv_mean = (wr[1] + wr[0]) / 2
        wv_half_width = (wr[1] - wr[0]) / 2

        # Generate S(lambda).
        wvs = np.linspace(wr[0], wr[1], 100)
        ss = np.exp(-0.5 * ((wvs - wv_mean) / wv_half_width)**2)

        return wvs, ss

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
