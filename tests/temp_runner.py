import numpy as np
import matplotlib.pyplot as plt

from exotic_ld import StellarLimbDarkening
from exotic_ld.ld_laws import quadratic_ld_law


sld = StellarLimbDarkening(
    M_H=-2.1, Teff=5101, logg=4.6, ld_model='stagger',
    ld_data_path='../data', interpolate_type="trilinear",
    custom_wavelengths=None, custom_mus=None,
    custom_stellar_model=None, verbose=True)
plt.plot(sld.stellar_wavelengths, sld.stellar_intensities[:, 0])
plt.plot(sld.stellar_wavelengths, sld.stellar_intensities[:, 3])
plt.plot(sld.stellar_wavelengths, sld.stellar_intensities[:, 6])
plt.xlabel('angstroms')
plt.xlim(0, 35000)
plt.show()


u1, u2 = sld.compute_quadratic_ld_coeffs(
    wavelength_range=[6000., 53000.], mode='JWST_NIRSpec_Prism')

print(u1, u2)
test_mu = np.linspace(0., 1., 100)
I_mu = quadratic_ld_law(test_mu, u1, u2)
plt.plot(test_mu, I_mu)
plt.xlabel('mu')
plt.show()

