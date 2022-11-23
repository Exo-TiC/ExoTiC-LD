import numpy as np
import matplotlib.pyplot as plt

from exotic_ld import StellarLimbDarkening
from exotic_ld.ld_laws import quadratic_ld_law


sld = StellarLimbDarkening(
    M_H=0.0, Teff=4500, logg=4.5, ld_model='1D',
    ld_data_path='../data')
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

