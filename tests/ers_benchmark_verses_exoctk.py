import matplotlib.pyplot as plt
import numpy as np

from exotic_ld import StellarLimbDarkening


M_H = 0.01
Teff = 5400.0
logg = 4.45
wr = np.array([28700, 51425.7])
m = 'JWST_NIRSpec_G395H'

sld = StellarLimbDarkening(M_H, Teff, logg, ld_model='1D',
                           ld_data_path='../data')

mus = np.linspace(0.05, 1, 1000)

# c1 = sld.compute_linear_ld_coeffs(wavelength_range=wr, mode=m)
# print(c1)
# plt.plot(mus, (1. - (0 * (1. - mus**0.5) + c1 * (1. - mus)
#             + 0 * (1. - mus**1.5) + 0 * (1. - mus**2))))
#
c1, c2 = sld.compute_quadratic_ld_coeffs(wavelength_range=wr, mode=m)
print(c1, c2)
plt.plot(mus, 1. - c1 * (1. - mus) - c2 * (1. - mus)**2)
plt.plot(mus, 1. - 0.063 * (1. - mus) - 0.194 * (1. - mus)**2)
#
# c1, c2, c3 = sld.compute_3_parameter_non_linear_ld_coeffs(wavelength_range=wr, mode=m)
# print(c1, c2, c3)
# plt.plot(mus, (1. - (0 * (1. - mus**0.5) + c1 * (1. - mus)
#             + c2 * (1. - mus**1.5) + c3 * (1. - mus**2))))

# c1, c2, c3, c4 = sld.compute_4_parameter_non_linear_ld_coeffs(wavelength_range=wr, mode=m)
# print(c1, c2, c3, c4)
# plt.plot(mus, (1. - (c1 * (1. - mus**0.5) + c2 * (1. - mus)
#             + c3 * (1. - mus**1.5) + c4 * (1. - mus**2))))
#
# plt.plot(mus, (1. - (0.037 * (1. - mus**0.5) + 0.64 * (1. - mus)
#             + -0.701 * (1. - mus**1.5) + 0.249 * (1. - mus**2))))

plt.show()
