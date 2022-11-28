import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import roots_legendre

from exotic_ld import StellarLimbDarkening


sld = StellarLimbDarkening(
    M_H=0.4, Teff=5000, logg=4.0, ld_model='mps1',
    ld_data_path='../../data', interpolate_type="nearest")
rs = (1 - sld.mus**2)**0.5

roots, weights = roots_legendre(500)
a, b = (0., 1.)
t = (b - a) / 2 * roots + (a + b) / 2

spectrum = []
for wv_idx in range(sld.stellar_wavelengths.shape[0]):

    i_interp_func = interp1d(
        rs, sld.stellar_intensities[wv_idx, :], kind='linear',
        bounds_error=False, fill_value=0.)

    def integrand(_r):
        return i_interp_func(_r) * _r * 2. * np.pi

    spectrum.append((b - a) / 2. * integrand(t).dot(weights))

plt.plot(sld.stellar_wavelengths, spectrum)
plt.xlim(0, 35000)
plt.show()
