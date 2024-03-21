import os
import numpy as np
import pandas as pd
import astropy.units as q
from astropy.io import fits
import astropy.constants as ac


# Define original data paths.
known_files_missing = []
overwrite = False
ld_data_path_original = '../exotic_ld_data_original'
stellar_data_path = os.path.join(ld_data_path_original, 'phoenix_v3_SpecIntFITS')
ld_data_path_new = '../../../data/exotic_ld_data'
new_data_path = os.path.join(ld_data_path_new, 'phoenix')

# Define parameter grids available.
M_H_grid = np.array([-4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.0, 0.5, 1.0])
Teff_grid = np.concatenate([np.arange(2300, 7000, 100), np.arange(7000, 12000, 200), np.arange(12000, 15001, 500)])
logg_grid = np.arange(0.0, 6.01, 0.5)


def read_phoenix_model(_M_H, _Teff, _logg):
    """
    Read in a phoenix model for a given M_H, Teff, and logg.

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
    t_str = "{0:05}".format(_Teff)
    lg_str = "{0:+.2f}".format(-_logg)
    mh_str = "{0:+.1f}".format(_M_H)
    # print((_Teff, _logg, _M_H), (t_str, lg_str, mh_str))
    file_name = "lte{}{}{}.PHOENIX-ACES-AGSS-COND-2011-SpecInt.fits".format(t_str, lg_str, mh_str)
    file_path = os.path.join(stellar_data_path, "Z{}".format(mh_str), file_name)

    try:
        hdul = fits.open(file_path)
        s_wvs = np.arange(hdul[0].header["CRVAL1"],
                          hdul[0].header["CRVAL1"] + hdul["PRIMARY"].data.shape[1] * hdul[0].header["CDELT1"],
                          hdul[0].header["CDELT1"])
        s_mus = hdul["MU"].data
        intensities = np.transpose(hdul["PRIMARY"].data, axes=(1, 0))
        hdul.close()
    except FileNotFoundError as err:
        return None, None, None
    except ValueError as err:
        print(file_name)
        raise err

    # Unpack the data and assign units.
    wavelengths = s_wvs * q.AA
    mu = np.flip(s_mus, axis=0)
    specific_intensity_wv = np.flip(intensities, axis=1) * q.erg / q.s / q.cm**2 / q.AA / q.steradian * 1e-8

    # Convert intensity from energy to number of photons.
    n_photon_intensity = specific_intensity_wv / (ac.h * ac.c / wavelengths[..., np.newaxis])

    # Update units [n_photons / s / cm^2 / Angstrom].
    n_photon_intensity = n_photon_intensity.to(1. / q.s / q.cm**2 / q.AA / q.steradian)

    return wavelengths, mu, n_photon_intensity


# Iterate models.
for M_H in M_H_grid:
    for Teff in Teff_grid:
        for logg in logg_grid:

            if M_H == -0.0:
                # Zero is negative in the phoenix dirs.
                # We make this positive.
                M_H_dir_str = 0.0
            else:
                M_H_dir_str = M_H

            # Create new dir.
            uld_dir = os.path.join(
                new_data_path, "MH{}".format(M_H_dir_str), "teff{}".format(Teff), "logg{}".format(logg))
            uld_path = os.path.join(uld_dir, "phoenix_spectra.dat")
            if not overwrite and os.path.exists(uld_path):
                print(Teff, logg, M_H, 'Already exists.')
                continue
            os.makedirs(uld_dir, exist_ok=True)

            # Read in stellar model.
            wv, mu, I_lambda_nu = read_phoenix_model(M_H, Teff, logg)

            if wv is None:
                # Model does not exist.
                continue

            # Write stellar model to uniform format.
            uld = pd.DataFrame(I_lambda_nu.value, index=wv.value, columns=mu)
            uld.to_csv(uld_path, index=True, header=True, sep=" ")
            with open(uld_path, "r+") as f:
                content = f.read()
                f.seek(0, 0)
                f.write("phoenix MH = {}  Teff = {} K  logg = {} v3.2\n".format(M_H, Teff, logg))
                f.write(content)
                print(Teff, logg, M_H, 'Created.')
