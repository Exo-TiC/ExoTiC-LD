import os
import numpy as np
from spectres import spectres
import matplotlib.pyplot as plt


# Stellar parameters.
M_H = float(0.0)
Teff = int(5500)
logg = float(4.5)

# Grids.
grids = ["phoenix", "kurucz", "mps1"]

# Plot each available spectrum.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
for g in grids:

    file_name = os.path.join(
        "../../../data/exotic_ld_data", g, "MH{}".format(M_H),
        "teff{}".format(Teff), "logg{}".format(logg),
        "{}_spectra.dat".format(g))
    print(file_name)

    try:
        mu_data = np.loadtxt(file_name, skiprows=1, max_rows=1)
        stellar_data = np.loadtxt(file_name, skiprows=2)
        print(stellar_data[:, 1:].shape)
    except FileNotFoundError as err:
        print('Model corresponding to Teff={}, logg={}, and M_H={} '
              'does not exist in the {} stellar models.'.format(
               Teff, logg, M_H, g))
        continue

    wvs_rebin = np.linspace(0, 35000, 1000)
    for mu in np.linspace(1, 0.1, 2):
        mu_idx = np.argmin(np.abs(mu_data - mu))
        ax1.plot(stellar_data[:, 0], stellar_data[:, 1 + mu_idx],
                 label="{} $\mu={}$".format(g, mu))
        ax2.plot(wvs_rebin, spectres(wvs_rebin, stellar_data[:, 0], stellar_data[:, 1 + mu_idx]))

ax1.set_xlim(0, 35000)
ax2.set_xlim(0, 35000)
ax1.set_xlabel('Wavelength / $\AA$')
ax2.set_xlabel('Wavelength / $\AA$')
ax1.set_ylabel('Intensity / $n_{\gamma} s^{-1} cm^{-2} {\AA}^{-1}$')
ax2.set_ylabel('Intensity / $n_{\gamma} s^{-1} cm^{-2} {\AA}^{-1}$')
ax1.legend()
plt.tight_layout()
plt.show()
