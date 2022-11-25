import unittest
import numpy as np

from exotic_ld import StellarLimbDarkening
from exotic_ld.ld_laws import linear_ld_law, quadratic_ld_law, \
    squareroot_ld_law, nonlinear_3param_ld_law, nonlinear_4param_ld_law


class TestLDC(unittest.TestCase):
    """ Test limb-darkening computation auto, no data available. """

    def __init__(self, *args, **kwargs):
        super(TestLDC, self).__init__(*args, **kwargs)

        # Make reproducible.
        np.random.seed(123)

        # Define custom/simulated modes.
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

        # Constants.
        self.h = 6.62607004e-34  # Planck's constant [SI].
        self.c = 2.99792458e8  # Speed of light [SI].
        self.k_b = 1.38064852e-23  # Boltzmann constant [SI].

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


if __name__ == '__main__':
    unittest.main()
