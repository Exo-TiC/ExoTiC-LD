import os
import numpy as np
import pandas as pd
import astropy.units as q
from scipy.io import readsav
import astropy.constants as ac


# Define original data paths.
overwrite = False
ld_data_path_original = '../data_original'
stellar_data_path = os.path.join(ld_data_path_original, 'Kurucz')
stellar_data_index = os.path.join(stellar_data_path, 'kuruczlist.sav')
ld_data_path_new = '../data'
new_data_path = os.path.join(ld_data_path_new, 'kurucz')

# Define parameter grids available.
M_H_grid = np.array(
    [-0.1, -0.2, -0.3, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0,
     -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0])
Teff_grid = np.array(
    [3500, 3750, 4000, 4250, 4500, 4750, 5000,
     5250, 5500, 5750, 6000, 6250, 6500])
logg_grid = np.array([4.0, 4.5, 5.0])

# Define corresponding model load positions.
M_H_grid_load = np.array(
    [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12,
     13, 14, 17, 20, 21, 22, 23, 24])
Teff_logg_grid_load = np.array(
    [[8, 19, 30, 41, 52, 63, 74, 85, 96, 107, 118, 129, 138],
     [9, 20, 31, 42, 53, 64, 75, 86, 97, 108, 119, 129, 139],
     [10, 21, 32, 43, 54, 65, 76, 87, 98, 109, 120, 130, 140]])


def read_kurucz_model(_M_H_idx, _M_H, _Teff_idx, _Teff, _logg_idx, _logg):
    """
    Read in a Kuruxz model for a given M_H, Teff, and logg.

    Inputs
    ======
        wavelength [nm]
        specific intensity [erg / s / cm^2 / Hz / steradian]
                            / I(mu=1) * 1e-5

    Return
    ======
        wavelength [angstroms].
        nu  [].
        photon_intensity  [n_photons / s / cm^2 / Angstrom / steradian].

    """
    idl_sf_list = readsav(stellar_data_index)
    stellar_model_name = bytes.decode(idl_sf_list['li'][M_H_grid_load[_M_H_idx]])
    load_number = Teff_logg_grid_load[_logg_idx][_Teff_idx]

    # Read in the stellar model data.
    n_header_rows = 3
    n_freq_intervals = 1221
    line_skip_data = (load_number + 1) * n_header_rows \
                     + load_number * n_freq_intervals
    try:
        stellar_data = pd.read_fwf(
            os.path.join(stellar_data_path, stellar_model_name),
            widths=[9, 10, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            header=None, skiprows=line_skip_data, nrows=n_freq_intervals)
    except FileNotFoundError as err:
        raise FileNotFoundError(
            'File {}, corresponding to Teff={}, logg={}, and M_H={} '
            'does not exist in the stellar models.'.format(
                stellar_model_name, _Teff, _logg, _M_H))

    # Unpack the data and assign units.
    wavelengths = (stellar_data.values[:, 0] * q.nm).to(q.AA)
    mu = np.array(
        [1.000, .900, .800, .700, .600, .500, .400, .300, .250,
         .200, .150, .125, .100, .075, .050, .025, .010])
    intensities = stellar_data.values[:, 1:]
    # NB. rescale values by mu=1 values as per...
    intensities[:, 1:] *= intensities[:, 0][..., np.newaxis]
    intensities[:, 1:] /= 1.e5
    specific_intensity_hz = intensities * q.erg / q.s / q.cm**2 / q.Hz / q.steradian

    # Convert intensity from per frequency to per wavelength.
    specific_intensity_wv = specific_intensity_hz * ac.c / wavelengths[..., np.newaxis]**2

    # Convert intensity from energy to number of photons.
    n_photon_intensity = specific_intensity_wv / (ac.h * ac.c / wavelengths[..., np.newaxis])

    # Update units [n_photons / s / cm^2 / Angstrom].
    n_photon_intensity = n_photon_intensity.to(1. / q.s / q.cm**2 / q.AA / q.steradian)

    return wavelengths, mu, n_photon_intensity


# Iterate models.
for Teff_idx, Teff in enumerate(Teff_grid):
    for logg_idx, logg in enumerate(logg_grid):
        for M_H_idx, M_H in enumerate(M_H_grid):

            # Create new dir.
            uld_dir = os.path.join(
                new_data_path, "MH{}".format(M_H), "teff{}".format(Teff), "logg{}".format(logg))
            uld_path = os.path.join(uld_dir, "kurucz_spectra.dat")
            if not overwrite and os.path.exists(uld_path):
                print(Teff, logg, M_H, 'Already exists.')
                continue
            os.makedirs(uld_dir, exist_ok=True)

            # Read in stellar model.
            wv, mu, I_lambda_nu = read_kurucz_model(
                M_H_idx, M_H, Teff_idx, Teff, logg_idx, logg)

            # Write stellar model to uniform format.
            uld = pd.DataFrame(I_lambda_nu, index=wv, columns=mu)
            uld.to_csv(uld_path, index=True, header=True, sep=" ")
            with open(uld_path, "r+") as f:
                content = f.read()
                f.seek(0, 0)
                f.write("kurucz MH = {}  Teff = {} K  logg = {}\n".format(M_H, Teff, logg))
                f.write(content)
                print(Teff, logg, M_H, 'Created.')
