import os
import shutil
import unittest
import requests
import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree

from exotic_ld.ld_grids import StellarGrids


class TestSM(unittest.TestCase):
    """ Test stellar models locally, with data downloaded. """

    @classmethod
    def setUpClass(cls):
        # Make reproducible.
        np.random.seed(123)

        # Define local test paths.
        cls.local_data_path = 'test_exotic_ld_data'

        # Define remote paths.
        cls.remote_ld_data_path = "https://www.star.bris.ac.uk/exotic-ld-data"
        cls.ld_data_version = ""

        # Clear cached data.
        if os.path.exists(cls.local_data_path):
            shutil.rmtree(cls.local_data_path)

        # Scaling parameters.
        cls._r_M_H = 1.00
        cls._r_Teff = 607.
        cls._r_logg = 1.54
        cls._r = np.array([cls._r_M_H, cls._r_Teff, cls._r_logg])

    @classmethod
    def tearDownClass(cls):
        # Tidy up.
        if os.path.exists(cls.local_data_path):
            shutil.rmtree(cls.local_data_path)

    def test_tree_pickles(self):
        """ Test tree pickles that hold stellar grid representation. """
        sg = StellarGrids(0., 0., 0., "phoenix", "", "", self.ld_data_version, "", 0)
        self._check_dimensions_and_range(sg, (5079, 3), (-1.5, 1.0), (2300, 15000), (0.0, 6.0))

        sg.ld_model = "kurucz"
        sg._get_stellar_model_kd_tree()
        self._check_dimensions_and_range(sg, (741, 3), (-5.0, 1.0), (3500, 6500), (4.0, 5.0))

        sg.ld_model = "stagger"
        sg._get_stellar_model_kd_tree()
        self._check_dimensions_and_range(sg, (99, 3), (-3.0, 0.0), (4000, 7000), (1.5, 5.0))

        sg.ld_model = "mps1"
        sg._get_stellar_model_kd_tree()
        self._check_dimensions_and_range(sg, (34160, 3), (-5.0, 1.5), (3500, 9000), (3., 5.))

        sg.ld_model = "mps2"
        sg._get_stellar_model_kd_tree()
        self._check_dimensions_and_range(sg, (34160, 3), (-5.0, 1.5), (3500, 9000), (3., 5.))

    def _check_dimensions_and_range(self, sg, e_dims, e_M_H_range, e_Teff_range, e_logg_range):
        self.assertIsInstance(sg._stellar_kd_tree, KDTree)
        self.assertEqual(sg._stellar_kd_tree.data.shape, e_dims)
        mins = sg._stellar_kd_tree.mins * self._r
        maxs = sg._stellar_kd_tree.maxes * self._r
        _M_H_range = (mins[0], maxs[0])
        _Teff_range = (int(round(mins[1])), int(round(maxs[1])))
        _logg_range = (mins[2], maxs[2])
        self.assertEqual(_M_H_range, e_M_H_range)
        self.assertEqual(_Teff_range, e_Teff_range)
        self.assertEqual(_logg_range, e_logg_range)

    def test_download_and_cache_exact_match(self):
        """ Test download and cache stellar data for an exact match. """
        _M_H = 0.5
        _Teff = 6000
        _logg = 4.
        ld_model = "kurucz"
        local_file_path = os.path.join(
            self.local_data_path, ld_model,
            "MH{}".format(str(round(_M_H, 2))),
            "teff{}".format(int(round(_Teff))),
            "logg{}".format(str(round(_logg, 1))),
            "{}_spectra{}.dat".format(ld_model, self.ld_data_version))
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
        self.assertFalse(os.path.exists(local_file_path))

        sg = StellarGrids(_M_H, _Teff, _logg, ld_model, self.local_data_path,
                          self.remote_ld_data_path, self.ld_data_version, "nearest", 1)
        wvs, mus, intensities = sg.get_stellar_data()

        self.assertTrue(os.path.exists(local_file_path))
        self.assertEqual(intensities.ndim, 2)
        self.assertEqual(wvs.shape[0], intensities.shape[0])
        self.assertEqual(mus.shape[0], intensities.shape[1])

    def test_download_and_cache_nearest(self):
        """ Test download and cache stellar data nearest matching. """
        _M_H = 0.51
        _Teff = 5091
        _logg = 4.11
        nearest_M_H = 0.5
        nearest_Teff = 5000
        nearest_logg = 4.0
        ld_model = "kurucz"
        local_file_path = os.path.join(
            self.local_data_path, ld_model,
            "MH{}".format(str(round(nearest_M_H, 2))),
            "teff{}".format(int(round(nearest_Teff))),
            "logg{}".format(str(round(nearest_logg, 1))),
            "{}_spectra{}.dat".format(ld_model, self.ld_data_version))
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
        self.assertFalse(os.path.exists(local_file_path))

        sg = StellarGrids(_M_H, _Teff, _logg, ld_model, self.local_data_path,
                          self.remote_ld_data_path, self.ld_data_version, "nearest", 1)
        wvs, mus, intensities = sg.get_stellar_data()

        self.assertTrue(os.path.exists(local_file_path))
        self.assertEqual(intensities.ndim, 2)
        self.assertEqual(wvs.shape[0], intensities.shape[0])
        self.assertEqual(mus.shape[0], intensities.shape[1])

    def test_download_and_cache_cuboid(self):
        """ Test download and cache stellar data for trilinear interpolation. """
        _M_H = 0.51
        _Teff = 5091
        _logg = 4.11
        cuboid_M_H = [0.5, 1.0]
        cuboid_Teff = [5000, 5250]
        cuboid_logg = [4.0, 4.5]
        ld_model = "kurucz"
        local_file_paths = []
        for c_M_H in cuboid_M_H:
            for c_Teff in cuboid_Teff:
                for c_logg in cuboid_logg:
                    local_file_path = os.path.join(
                        self.local_data_path, ld_model,
                        "MH{}".format(str(round(c_M_H, 2))),
                        "teff{}".format(int(round(c_Teff))),
                        "logg{}".format(str(round(c_logg, 1))),
                        "{}_spectra{}.dat".format(ld_model, self.ld_data_version))
                    if os.path.exists(local_file_path):
                        os.remove(local_file_path)
                    self.assertFalse(os.path.exists(local_file_path))
                    local_file_paths.append(local_file_path)

        sg = StellarGrids(_M_H, _Teff, _logg, ld_model, self.local_data_path,
                          self.remote_ld_data_path, self.ld_data_version, "trilinear", 1)
        wvs, mus, intensities = sg.get_stellar_data()

        for local_file_path in local_file_paths:
            self.assertTrue(os.path.exists(local_file_path))
        self.assertEqual(intensities.ndim, 2)
        self.assertEqual(wvs.shape[0], intensities.shape[0])
        self.assertEqual(mus.shape[0], intensities.shape[1])

    @unittest.skip("No url checks.")
    def test_phoenix_urls(self):
        """ Test Phoenix urls exist. """
        sg = StellarGrids(0., 0., 0., "phoenix", "", "", self.ld_data_version, "", 0)
        with tqdm(total=sg._stellar_kd_tree.data.shape[0]) as pbar:
            for _params in sg._stellar_kd_tree.data:
                r_params = _params * self._r
                r_params[0] = 0.0 if r_params[0] == -0.0 else r_params[0]  # Make zeros not negative.
                remote_file_path = os.path.join(
                    self.remote_ld_data_path, sg.ld_model,
                    "MH{}".format(str(round(r_params[0], 2))),
                    "teff{}".format(int(round(r_params[1]))),
                    "logg{}".format(str(round(r_params[2], 1))),
                    "{}_spectra{}.dat".format(sg.ld_model, sg._ld_data_version))
                response = requests.head(remote_file_path)
                response.raise_for_status()
                pbar.update(1)

    @unittest.skip("No url checks.")
    def test_kurucz_urls(self):
        """ Test kurucz urls exist. """
        sg = StellarGrids(0., 0., 0., "kurucz", "", "", self.ld_data_version, "", 0)
        with tqdm(total=sg._stellar_kd_tree.data.shape[0]) as pbar:
            for _params in sg._stellar_kd_tree.data:
                r_params = _params * self._r
                remote_file_path = os.path.join(
                    self.remote_ld_data_path, sg.ld_model,
                    "MH{}".format(str(round(r_params[0], 2))),
                    "teff{}".format(int(round(r_params[1]))),
                    "logg{}".format(str(round(r_params[2], 1))),
                    "{}_spectra{}.dat".format(sg.ld_model, sg._ld_data_version))
                response = requests.head(remote_file_path)
                response.raise_for_status()
                pbar.update(1)

    @unittest.skip("No url checks.")
    def test_stagger_urls(self):
        """ Test stagger urls exist. """
        sg = StellarGrids(0., 0., 0., "stagger", "", "", self.ld_data_version, "", 0)
        with tqdm(total=sg._stellar_kd_tree.data.shape[0]) as pbar:
            for _params in sg._stellar_kd_tree.data:
                r_params = _params * self._r
                remote_file_path = os.path.join(
                    self.remote_ld_data_path, sg.ld_model,
                    "MH{}".format(str(round(r_params[0], 2))),
                    "teff{}".format(int(round(r_params[1]))),
                    "logg{}".format(str(round(r_params[2], 1))),
                    "{}_spectra{}.dat".format(sg.ld_model, sg._ld_data_version))
                response = requests.head(remote_file_path)
                response.raise_for_status()
                pbar.update(1)

    @unittest.skip("No url checks.")
    def test_mps1_urls(self):
        """ Test mps1 urls exist. """
        sg = StellarGrids(0., 0., 0., "mps1", "", "", self.ld_data_version, "", 0)
        with tqdm(total=sg._stellar_kd_tree.data.shape[0]) as pbar:
            for _params in sg._stellar_kd_tree.data:
                r_params = _params * self._r
                remote_file_path = os.path.join(
                    self.remote_ld_data_path, sg.ld_model,
                    "MH{}".format(str(round(r_params[0], 2))),
                    "teff{}".format(int(round(r_params[1]))),
                    "logg{}".format(str(round(r_params[2], 1))),
                    "{}_spectra{}.dat".format(sg.ld_model, sg._ld_data_version))
                response = requests.head(remote_file_path)
                response.raise_for_status()
                pbar.update(1)

    @unittest.skip("No url checks.")
    def test_mps2_urls(self):
        """ Test mps2 urls exist. """
        sg = StellarGrids(0., 0., 0., "mps2", "", "", self.ld_data_version, "", 0)
        with tqdm(total=sg._stellar_kd_tree.data.shape[0]) as pbar:
            for _params in sg._stellar_kd_tree.data:
                r_params = _params * self._r
                remote_file_path = os.path.join(
                    self.remote_ld_data_path, sg.ld_model,
                    "MH{}".format(str(round(r_params[0], 2))),
                    "teff{}".format(int(round(r_params[1]))),
                    "logg{}".format(str(round(r_params[2], 1))),
                    "{}_spectra{}.dat".format(sg.ld_model, sg._ld_data_version))
                response = requests.head(remote_file_path)
                response.raise_for_status()
                pbar.update(1)


if __name__ == '__main__':
    unittest.main()
