import os
import numpy as np
import pandas as pd
import astropy.units as q
from scipy.io import readsav
import astropy.constants as ac


# Define original data paths.
overwrite = False
ld_data_path_original = '../data_original'
stellar_data_path = os.path.join(ld_data_path_original, '3DGrid')
ld_data_path_new = '../data'
new_data_path = os.path.join(ld_data_path_new, 'stagger')

# Define parameter grids available, {Teff: {logg}: [M_H]}.
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


# Define corresponding model load positions.
M_H_grid_load = {-3.0: '30', -2.0: '20', -1.0: '10', 0.0: '00'}
Teff_grid_load = {4000: '40', 4500: '45', 5000: '50', 5500: '55',
                  5777: '5777', 6000: '60', 6500: '65', 7000: '70'}
logg_grid_load = {1.5: '15', 2.0: '20', 2.5: '25', 3.0: '30',
                  3.5: '35', 4.0: '40', 4.4: '44', 4.5: '45', 5.0: '50'}


def read_stagger_model(_M_H, _Teff, _logg):
    """
    Read in a Stagger model for a given M_H, Teff, and logg.

    Inputs
    ======
        wavelength [Angstroms]
        specific intensity [erg / s / cm^2 / Angstroms / steradian * 1e-2]

    Return
    ======
        wavelength [angstroms].
        nu  [].
        photon_intensity  [n_photons / s / cm^2 / Angstrom].

    """
    load_file = 'mmu_t' + Teff_grid_load[_Teff] \
                + 'g' + logg_grid_load[_logg] \
                + 'm' + M_H_grid_load[_M_H] + 'v05.flx'

    try:
        sav = readsav(os.path.join(stellar_data_path, load_file))
    except FileNotFoundError as err:
        raise FileNotFoundError(
            'File {}, corresponding to Teff={}, logg={}, and M_H={} '
            'does not exist in the stellar models.'.format(
                load_file, _Teff, _logg, _M_H))

    # Unpack the data and assign units.
    wavelengths = sav['mmd'].lam[0] * q.AA
    mu = sav['mmd'].mu
    intensities = np.array(sav['mmd'].flx.tolist()).T

    # Replace nans.
    intensities[np.isnan(intensities)] = 0.

    # Flip data so that mu decreases from left to right.
    # And drop mu=0. point.
    mu = np.flip(mu, axis=0)[:-1]
    intensities = np.flip(intensities, axis=1)[:, :-1]

    specific_intensity_wv = intensities * q.erg / q.s / q.cm**2 / q.AA / q.steradian * 1e-2

    # Convert intensity from energy to number of photons.
    n_photon_intensity = specific_intensity_wv / (ac.h * ac.c / wavelengths[..., np.newaxis])

    # Update units [n_photons / s / cm^2 / Angstrom].
    n_photon_intensity = n_photon_intensity.to(1. / q.s / q.cm**2 / q.AA / q.steradian)

    return wavelengths, mu, n_photon_intensity


# Iterate models.
for Teff, logg_dict in whole_grid.items():
    for logg, M_H_list in logg_dict.items():
        for M_H in M_H_list:

            # Create new dir.
            uld_dir = os.path.join(
                new_data_path, "MH{}".format(M_H), "teff{}".format(Teff), "logg{}".format(logg))
            uld_path = os.path.join(uld_dir, "stagger_spectra.dat")
            if not overwrite and os.path.exists(uld_path):
                print(Teff, logg, M_H, 'Already exists.')
                continue
            os.makedirs(uld_dir, exist_ok=True)

            # Read in stellar model.
            wv, mu, I_lambda_nu = read_stagger_model(M_H, Teff, logg)

            # Write stellar model to uniform format.
            uld = pd.DataFrame(I_lambda_nu, index=wv, columns=mu)
            uld.to_csv(uld_path, index=True, header=True, sep=" ")
            with open(uld_path, "r+") as f:
                content = f.read()
                f.seek(0, 0)
                f.write("stagger MH = {}  Teff = {} K  logg = {}\n".format(M_H, Teff, logg))
                f.write(content)
                print(Teff, logg, M_H, 'Created.')
