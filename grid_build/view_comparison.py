import os
import numpy as np
import matplotlib.pyplot as plt


# Stellar parameters.
M_H = 0.4
Teff = 5000
logg = 4.2

# Grids.
grids = ['kurucz', 'stagger', 'mps1', 'mps2']

# Plot each available spectrum.
for g in grids:

    file_name = os.path.join(
        "../data", g, "MH{}".format(M_H),
        "teff{}".format(Teff), "logg{}".format(logg),
        "{}_spectra.dat".format(g))

    try:
        mu_data = np.loadtxt(file_name, skiprows=1, max_rows=1)
        stellar_data = np.loadtxt(file_name, skiprows=2)
    except FileNotFoundError as err:
        print('Model corresponding to Teff={}, logg={}, and M_H={} '
              'does not exist in the {} stellar models.'.format(
               Teff, logg, M_H, g))
        continue

    for mu in np.linspace(1, 0, 5):
        mu_idx = np.argmin(np.abs(mu_data - mu))
        plt.plot(stellar_data[:, 0], stellar_data[:, 1 + mu_idx],
                 label="{} $\mu={}$".format(g, mu))

    plt.xlim(0, 35000)
    plt.legend()
    plt.xlabel('Wavelength / $\AA$')
    plt.ylabel('Intensity / $n_{\gamma} s^{-1} cm^{-2} {\AA}^{-1}$')
    plt.show()
