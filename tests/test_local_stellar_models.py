import unittest
import numpy as np
from tqdm import tqdm

from exotic_ld import StellarLimbDarkening


class TestLDC(unittest.TestCase):
    """ Test stellar models locally, with data downloaded. """

    def __init__(self, *args, **kwargs):
        super(TestLDC, self).__init__(*args, **kwargs)

        # Make reproducible.
        np.random.seed(123)

        # Define local test paths.
        self.local_data_path = '../../../data/exotic_ld_data'

    def test_kurucz_grid(self):
        """ Test Kurucz grid. """
        M_H_grid = np.array(
            [-0.1, -0.2, -0.3, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0,
             -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0])
        Teff_grid = np.array(
            [3500, 3750, 4000, 4250, 4500, 4750, 5000,
             5250, 5500, 5750, 6000, 6250, 6500])
        logg_grid = np.array([4.0, 4.5, 5.0])

        n_expected = 741
        n_checked = 0
        with tqdm(total=n_expected) as pbar:
            for Teff in Teff_grid:
                for logg in logg_grid:
                    for M_H in M_H_grid:
                        StellarLimbDarkening(
                            M_H=M_H, Teff=Teff, logg=logg, ld_model='kurucz',
                            ld_data_path=self.local_data_path)
                        n_checked += 1
                        pbar.update(1)
        self.assertEqual(n_expected, n_checked)

    def test_stagger_grid(self):
        """ Test Stagger grid. """
        whole_grid = {4000: {1.5: [-3.0, -2.0, -1.0, 0.0],
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

        n_expected = 99
        n_checked = 0
        with tqdm(total=n_expected) as pbar:
            for Teff, logg_dict in whole_grid.items():
                for logg, M_H_list in logg_dict.items():
                    for M_H in M_H_list:
                        StellarLimbDarkening(
                            M_H=M_H, Teff=Teff, logg=logg, ld_model='stagger',
                            ld_data_path=self.local_data_path)
                        n_checked += 1
                        pbar.update(1)
        self.assertEqual(n_expected, n_checked)

    def test_mps1_grid(self):
        """ Test MPS-ATLAS-1 grid. """
        M_H_grid = np.array(
            [-0.1, -0.2, -0.3, -0.4, -0.5, -0.05, -0.6, -0.7, -0.8, -0.9,
             -0.15, -0.25, -0.35, -0.45, -0.55, -0.65, -0.75, -0.85, -0.95,
             -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, -1.9,
             -2.0, -2.1, -2.2, -2.3, -2.4, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0,
             0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.05, 0.6, 0.7, 0.8, 0.9,
             0.15, 0.25, 0.35, 0.45, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        Teff_grid = np.arange(3500, 9050, 100)
        logg_grid = np.array([3.0, 3.5, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.0])

        n_expected = 34160
        n_checked = 0
        with tqdm(total=n_expected) as pbar:
            for Teff in Teff_grid:
                for logg in logg_grid:
                    for M_H in M_H_grid:
                        StellarLimbDarkening(
                            M_H=M_H, Teff=Teff, logg=logg, ld_model='mps1',
                            ld_data_path=self.local_data_path)
                        n_checked += 1
                        pbar.update(1)
        self.assertEqual(n_expected, n_checked)

    def test_mps2_grid(self):
        """ Test MPS-ATLAS-2 grid. """
        M_H_grid = np.array(
            [-0.1, -0.2, -0.3, -0.4, -0.5, -0.05, -0.6, -0.7, -0.8, -0.9,
             -0.15, -0.25, -0.35, -0.45, -0.55, -0.65, -0.75, -0.85, -0.95,
             -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, -1.9,
             -2.0, -2.1, -2.2, -2.3, -2.4, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0,
             0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.05, 0.6, 0.7, 0.8, 0.9,
             0.15, 0.25, 0.35, 0.45, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        Teff_grid = np.arange(3500, 9050, 100)
        logg_grid = np.array([3.0, 3.5, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.0])

        n_expected = 34160
        n_checked = 0
        with tqdm(total=n_expected) as pbar:
            for Teff in Teff_grid:
                for logg in logg_grid:
                    for M_H in M_H_grid:
                        StellarLimbDarkening(
                            M_H=M_H, Teff=Teff, logg=logg, ld_model='mps2',
                            ld_data_path=self.local_data_path)
                        n_checked += 1
                        pbar.update(1)
        self.assertEqual(n_expected, n_checked)

    def test_grid_interpolation(self):
        """ Test all grids, spot check interpolation. """
        for ld_model in ['kurucz', 'stagger', 'mps1', 'mps2']:
            # Nearest.
            sld = StellarLimbDarkening(
                M_H=0.01, Teff=4520, logg=4.88, ld_model=ld_model,
                ld_data_path=self.local_data_path, interpolate_type="nearest")
            self.assertEqual((sld.stellar_wavelengths.shape[0],
                              sld.mus.shape[0]), sld.stellar_intensities.shape)

            # Interp one parameter.
            sld = StellarLimbDarkening(
                M_H=0.0, Teff=4500, logg=4.88, ld_model=ld_model,
                ld_data_path=self.local_data_path, interpolate_type="trilinear")
            self.assertEqual((sld.stellar_wavelengths.shape[0],
                              sld.mus.shape[0]), sld.stellar_intensities.shape)

            # Interp two parameters.
            sld = StellarLimbDarkening(
                M_H=0.0, Teff=4520, logg=4.88, ld_model=ld_model,
                ld_data_path=self.local_data_path, interpolate_type="trilinear")
            self.assertEqual((sld.stellar_wavelengths.shape[0],
                              sld.mus.shape[0]), sld.stellar_intensities.shape)

            # Interp all three parameters.
            sld = StellarLimbDarkening(
                M_H=0.01, Teff=4520, logg=4.88, ld_model=ld_model,
                ld_data_path=self.local_data_path, interpolate_type="trilinear")
            self.assertEqual((sld.stellar_wavelengths.shape[0],
                              sld.mus.shape[0]), sld.stellar_intensities.shape)

            # Parameter beyond range.
            sld = StellarLimbDarkening(
                M_H=50.01, Teff=4520, logg=4.88, ld_model=ld_model,
                ld_data_path=self.local_data_path, interpolate_type="trilinear")
            self.assertEqual((sld.stellar_wavelengths.shape[0],
                              sld.mus.shape[0]), sld.stellar_intensities.shape)


if __name__ == '__main__':
    unittest.main()
