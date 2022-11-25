import numpy as np
import matplotlib.pyplot as plt

from exotic_ld import StellarLimbDarkening
from exotic_ld.ld_laws import linear_ld_law, quadratic_ld_law, \
    squareroot_ld_law, nonlinear_3param_ld_law, nonlinear_4param_ld_law


sld = StellarLimbDarkening(
    M_H=0.0, Teff=3500, logg=4.5, ld_model='1D',
    ld_data_path='../data', interpolate_type="nearest",
    custom_wavelengths=None, custom_mus=None,
    custom_stellar_model=None, verbose=True)

# plt.plot(sld.stellar_wavelengths, sld.stellar_intensities[:, 0])
# plt.plot(sld.stellar_wavelengths, sld.stellar_intensities[:, 3])
# plt.plot(sld.stellar_wavelengths, sld.stellar_intensities[:, 6])
# plt.xlabel('angstroms')
# plt.xlim(0, 35000)
# plt.show()

test_mu = np.linspace(0., 1., 100)

u1, u1_sigma = sld.compute_linear_ld_coeffs(
    wavelength_range=[9000., 10000.], mode='JWST_NIRSpec_Prism',
    mu_min=0.1, return_sigmas=True)

u1_dist = np.random.normal(loc=u1, scale=u1_sigma, size=1000)
I_mu = linear_ld_law(test_mu[..., np.newaxis], u1_dist)
I_mu_16p = np.percentile(I_mu, 16., axis=1)
I_mu_50p = np.percentile(I_mu, 50., axis=1)
I_mu_84p = np.percentile(I_mu, 84., axis=1)
plt.scatter(sld.mus, sld.I_mu)
plt.plot(test_mu, I_mu_50p)
plt.fill_between(test_mu, I_mu_16p, I_mu_84p)
plt.xlabel('mu')
plt.ylim(0, 1.1)
plt.show()

u1, u1_sigma = sld.compute_linear_ld_coeffs(
    wavelength_range=[50000., 51000.], mode='JWST_NIRSpec_Prism',
    mu_min=0.1, return_sigmas=True)

u1_dist = np.random.normal(loc=u1, scale=u1_sigma, size=1000)
I_mu = linear_ld_law(test_mu[..., np.newaxis], u1_dist)
I_mu_16p = np.percentile(I_mu, 16., axis=1)
I_mu_50p = np.percentile(I_mu, 50., axis=1)
I_mu_84p = np.percentile(I_mu, 84., axis=1)
plt.scatter(sld.mus, sld.I_mu)
plt.plot(test_mu, I_mu_50p)
plt.fill_between(test_mu, I_mu_16p, I_mu_84p)
plt.xlabel('mu')
plt.ylim(0, 1.1)
plt.show()

us, us_sigmas = sld.compute_quadratic_ld_coeffs(
    wavelength_range=[9000., 10000.], mode='JWST_NIRSpec_Prism',
    mu_min=0.1, return_sigmas=True)

us_dist = np.random.normal(loc=us, scale=us_sigmas, size=(1000, len(us)))
I_mu = quadratic_ld_law(test_mu[..., np.newaxis], us_dist[:, 0], us_dist[:, 1])
I_mu_percentiles = np.percentile(I_mu, [16., 50., 84.], axis=1)

plt.scatter(sld.mus, sld.I_mu)
plt.plot(test_mu, I_mu_percentiles[1])
plt.fill_between(test_mu, I_mu_percentiles[0], I_mu_percentiles[2], alpha=0.3)
plt.xlabel('mu')
plt.ylim(0, 1.1)
plt.show()

us, us_sigmas = sld.compute_4_parameter_non_linear_ld_coeffs(
    wavelength_range=[9000., 10000.], mode='JWST_NIRSpec_Prism',
    mu_min=0.1, return_sigmas=True)

us_dist = np.random.normal(loc=us, scale=us_sigmas, size=(1000, len(us)))
I_mu = nonlinear_4param_ld_law(test_mu[..., np.newaxis],
                               us_dist[:, 0], us_dist[:, 1],
                               us_dist[:, 2], us_dist[:, 3])
I_mu_percentiles = np.percentile(I_mu, [16., 50., 84.], axis=1)

plt.scatter(sld.mus, sld.I_mu)
plt.plot(test_mu, I_mu_percentiles[1])
plt.fill_between(test_mu, I_mu_percentiles[0], I_mu_percentiles[2], alpha=0.3)
plt.xlabel('mu')
plt.ylim(0, 1.1)
plt.show()

