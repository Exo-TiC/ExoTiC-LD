import os
import numpy as np
import pandas as pd
import astropy.units as q
import astropy.constants as ac


# Define original data paths.
known_files_missing = []
overwrite = False
ld_data_path_original = '../data_original'
stellar_data_path = os.path.join(ld_data_path_original, 'MPS-ATLAS-1')
ld_data_path_new = '../data'
new_data_path = os.path.join(ld_data_path_new, 'mps1')

# Define parameter grids available.
M_H_grid = np.array(
    [-0.1, -0.2, -0.3, -0.4, -0.5, -0.05, -0.6, -0.7, -0.8, -0.9,
     -0.15, -0.25, -0.35, -0.45, -0.55, -0.65, -0.75, -0.85, -0.95,
     -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, -1.9,
     -2.0, -2.1, -2.2, -2.3, -2.4, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0,
     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.05, 0.6, 0.7, 0.8, 0.9,
     0.15, 0.25, 0.35, 0.45, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
Teff_grid = np.arange(3500, 9050, 100)
logg_grid = np.array([3.0, 3.5, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.0])


def read_mps_atlas_1_model(_M_H, _Teff, _logg):
    """
    Read in an MPS-ATLAS-1 model for a given M_H, Teff, and logg.

    Inputs
    ======
        wavelength [nm]
        specific intensity [erg / s / cm^2 / Hz / steradian]

    Return
    ======
        wavelength [angstroms].
        nu  [].
        photon_intensity  [n_photons / s / cm^2 / Angstrom / steradian].

    """
    file_name = os.path.join(stellar_data_path, "MH{}".format(_M_H),
                             "teff{}".format(_Teff), "logg{}".format(_logg),
                             "mpsa_intensity_spectra.dat")
    try:
        stellar_data = np.loadtxt(file_name, skiprows=2)
    except FileNotFoundError as err:
        if file_name in known_files_missing:
            return None, None, None
        else:
            raise FileNotFoundError(
                'File {}, corresponding to Teff={}, logg={}, and M_H={} '
                'does not exist in the stellar models.'.format(
                    file_name, _Teff, _logg, _M_H))
    except ValueError as err:
        print(file_name)
        raise err

    # Unpack the data and assign units.
    wavelengths = (stellar_data[:, 0] * q.nm).to(q.AA)
    mu = np.array(
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.30, 0.25, 0.22,
         0.20, 0.17, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05,
         0.03, 0.02, 0.01])
    specific_intensity_hz = stellar_data[:, 1:] \
                            * q.erg / q.s / q.cm**2 / q.Hz / q.steradian

    # Convert intensity from per frequency to per wavelength.
    specific_intensity_wv = specific_intensity_hz * ac.c / wavelengths[..., np.newaxis]**2

    # Convert intensity from energy to number of photons.
    n_photon_intensity = specific_intensity_wv / (ac.h * ac.c / wavelengths[..., np.newaxis])

    # Update units [n_photons / s / cm^2 / Angstrom].
    n_photon_intensity = n_photon_intensity.to(1. / q.s / q.cm**2 / q.AA / q.steradian)

    return wavelengths, mu, n_photon_intensity


# Iterate models.
for Teff in Teff_grid:
    for logg in logg_grid:
        for M_H in M_H_grid:

            # Create new dir.
            uld_dir = os.path.join(
                new_data_path, "MH{}".format(M_H), "teff{}".format(Teff), "logg{}".format(logg))
            uld_path = os.path.join(uld_dir, "mps1_spectra.dat")
            if not overwrite and os.path.exists(uld_path):
                print(Teff, logg, M_H, 'Already exists.')
                continue
            os.makedirs(uld_dir, exist_ok=True)

            # Read in stellar model.
            wv, mu, I_lambda_nu = read_mps_atlas_1_model(M_H, Teff, logg)

            # Write stellar model to uniform format.
            uld = pd.DataFrame(I_lambda_nu, index=wv, columns=mu)
            uld.to_csv(uld_path, index=True, header=True, sep=" ")
            with open(uld_path, "r+") as f:
                content = f.read()
                f.seek(0, 0)
                f.write("mps-atlas-1 MH = {}  Teff = {} K  logg = {}\n".format(M_H, Teff, logg))
                f.write(content)
                print(Teff, logg, M_H, 'Created.')
