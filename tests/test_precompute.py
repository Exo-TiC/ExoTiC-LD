import os
import shutil
import unittest
import numpy as np
from tqdm import tqdm

from exotic_ld import StellarLimbDarkening, PrecomputedLimbDarkening


class TestPLD(unittest.TestCase):
    """Test stellar models locally, with data downloaded."""

    @classmethod
    def setUpClass(cls):
        # Make reproducible.
        np.random.seed(123)

        # Define local test paths.
        cls.local_data_path = "test_exotic_ld_data"

        # Define remote paths.
        cls.remote_ld_data_path = "https://www.star.bris.ac.uk/exotic-ld-data"
        cls.ld_data_version = ""

        # Clear cached data.
        if os.path.exists(cls.local_data_path):
            shutil.rmtree(cls.local_data_path)


    @classmethod
    def tearDownClass(cls):
        # Tidy up.
        if os.path.exists(cls.local_data_path):
            shutil.rmtree(cls.local_data_path)

    def test_build_grid_supported_mode(self):
        pld = PrecomputedLimbDarkening(
            ld_model="kurucz",
            ld_data_path=self.local_data_path,
            mode="JWST_NIRSpec_Prism",
            wavelength_range=None,
            custom_wavelengths=None,
            custom_throughput=None,
            interpolate_type="nearest",
            ld_data_version="3.2.0",
            verbose=-1,
            save_stellar_grid=None,
        )
        local_file_path = pld.cache_file
        self.assertTrue(os.path.exists(local_file_path))

    def test_build_grid_custom_mode(self):
        waves = np.linspace(4_000, 10_000, 100)
        throughput = np.ones_like(waves)
        pld = PrecomputedLimbDarkening(
            ld_model="kurucz",
            ld_data_path=self.local_data_path,
            mode="custom",
            wavelength_range=None,
            custom_wavelengths=waves,
            custom_throughput=throughput,
            interpolate_type="nearest",
            ld_data_version="3.2.0",
            verbose=-1,
            save_stellar_grid=None,
        )
        local_file_path = pld.cache_file
        self.assertTrue(os.path.exists(local_file_path))

    def test_agreement_with_sld_supported_mode(self):
        # make sure that when using interpolation_type="nearest",
        # the precomputed limb darkening model agrees with the stellar limb darkening model

        np.random.seed(0)

        # load the precomputed limb darkening model
        pld = PrecomputedLimbDarkening(
            ld_model="kurucz",
            ld_data_path=self.local_data_path,
            mode="JWST_NIRSpec_Prism",
            wavelength_range=None,
            custom_wavelengths=None,
            custom_throughput=None,
            interpolate_type="nearest",
            ld_data_version="3.2.0",
            verbose=-1,
            save_stellar_grid=None,
        )

        pld_laws = [
            pld.compute_linear_ld_coeffs,
            pld.compute_quadratic_ld_coeffs,
            pld.compute_kipping_ld_coeffs,
            pld.compute_squareroot_ld_coeffs,
            pld.compute_3_parameter_non_linear_ld_coeffs,
            pld.compute_4_parameter_non_linear_ld_coeffs,
        ]

        # loop through 1000 random stellar parameters
        for i in tqdm(range(1000)):
            m_h = np.random.uniform(-5, 1)
            teff = np.random.uniform(3500, 6500)
            logg = np.random.uniform(3, 5)

            sld = StellarLimbDarkening(
                M_H=m_h,
                Teff=teff,
                logg=logg,
                ld_model="kurucz",
                ld_data_path=self.local_data_path,
                interpolate_type="nearest",
                verbose=-1,
            )

            args = {
                "wavelength_range": [6_000.0, 53_000.0],
                "mode": "JWST_NIRSpec_Prism",
                "custom_wavelengths": None,
                "custom_throughput": None,
                "return_sigmas": True,
            }

            sld_laws = [
                sld.compute_linear_ld_coeffs,
                sld.compute_quadratic_ld_coeffs,
                sld.compute_kipping_ld_coeffs,
                sld.compute_squareroot_ld_coeffs,
                sld.compute_3_parameter_non_linear_ld_coeffs,
                sld.compute_4_parameter_non_linear_ld_coeffs,
            ]

            # loop through all of the limb darkening laws, making sure the
            # coefficients and sigmas are the same
            for j in range(len(sld_laws)):
                s_coeffs = sld_laws[j](**args)
                p_coeffs = pld_laws[j](
                    M_H=m_h, Teff=teff, logg=logg, return_sigmas=True
                )
                self.assertTrue(s_coeffs == p_coeffs)

    def test_agreement_with_sld_custom_mode(self):
        # same as above but with the custom mode
        np.random.seed(0)

        waves = np.linspace(4_000, 10_000, 100)
        throughput = np.ones_like(waves)
        pld = PrecomputedLimbDarkening(
            ld_model="kurucz",
            ld_data_path=self.local_data_path,
            mode="custom",
            wavelength_range=None,
            custom_wavelengths=waves,
            custom_throughput=throughput,
            interpolate_type="nearest",
            ld_data_version="3.2.0",
            verbose=0,
            save_stellar_grid=None,
        )

        pld_laws = [
            pld.compute_linear_ld_coeffs,
            pld.compute_quadratic_ld_coeffs,
            pld.compute_kipping_ld_coeffs,
            pld.compute_squareroot_ld_coeffs,
            pld.compute_3_parameter_non_linear_ld_coeffs,
            pld.compute_4_parameter_non_linear_ld_coeffs,
        ]

        for i in tqdm(range(1000)):
            m_h = np.random.uniform(-5, 1)
            teff = np.random.uniform(3500, 6500)
            logg = np.random.uniform(3, 5)

            sld = StellarLimbDarkening(
                M_H=m_h,
                Teff=teff,
                logg=logg,
                ld_model="kurucz",
                ld_data_path=self.local_data_path,
                interpolate_type="nearest",
                verbose=-1,
            )

            args = {
                "wavelength_range": [4_000.0, 10_000.0],
                "mode": "custom",
                "custom_wavelengths": waves,
                "custom_throughput": throughput,
                "return_sigmas": True,
            }

            sld_laws = [
                sld.compute_linear_ld_coeffs,
                sld.compute_quadratic_ld_coeffs,
                sld.compute_kipping_ld_coeffs,
                sld.compute_squareroot_ld_coeffs,
                sld.compute_3_parameter_non_linear_ld_coeffs,
                sld.compute_4_parameter_non_linear_ld_coeffs,
            ]

            for j in range(len(sld_laws)):
                s_coeffs = sld_laws[j](**args)
                p_coeffs = pld_laws[j](
                    M_H=m_h, Teff=teff, logg=logg, return_sigmas=True
                )
                assert s_coeffs == p_coeffs
