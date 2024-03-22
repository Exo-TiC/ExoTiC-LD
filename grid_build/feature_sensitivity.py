import os
import numpy as np
import matplotlib.pyplot as plt

from exotic_ld import StellarLimbDarkening


def l1_losses(_M_Hs, _Teffs, _loggs):
    """ Compute list of L1 losses[I(mus, x), I(mus, x + delta_x)]. """
    delta_I_mu = []
    prev_I_mu = None
    for _M_H in _M_Hs:
        for _Teff in _Teffs:
            for _logg in _loggs:
                sld = StellarLimbDarkening(M_H=_M_H, Teff=_Teff, logg=_logg, ld_model="mps1",
                                           ld_data_path=os.environ["exotic_ld_data"],
                                           interpolate_type="nearest", verbose=True)
                sld.compute_4_parameter_non_linear_ld_coeffs(wavelength_range=[10000., 40000.],
                                                             mode="JWST_NIRSpec_prism", mu_min=0.1)
                if prev_I_mu is not None:
                    l1_loss = np.sum(np.abs(sld.I_mu - prev_I_mu))
                    delta_I_mu.append(l1_loss)
                prev_I_mu = sld.I_mu

    return delta_I_mu


# Check sensitivity of I(mu) to  Delta Teff, at a number of M_H, logg points.
delta_I_mu_Teff = []
Teffs = np.sort(np.arange(3500, 9050, 100))
for M_H in [-1., 0., 1.]:
    for logg in [3.0, 4.0, 5.0]:
        delta_I_mu_Teff.append(l1_losses([M_H], Teffs, [logg]))

# Check sensitivity of I(mu) to  Delta logg, at a number of M_H, Teff points.
delta_I_mu_logg = []
loggs = np.sort([3.0, 3.5, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.0])
for M_H in [-1., 0., 1.]:
    for Teff in [3500, 6000, 9000]:
        delta_I_mu_logg.append(l1_losses([M_H], [Teff], loggs))

# Check sensitivity of I(mu) to Delta M_H, at a number of Teff, logg points.
delta_I_mu_M_H = []
M_Hs = np.sort([-0.1, -0.2, -0.3, -0.4, -0.5, -0.05, -0.6, -0.7, -0.8, -0.9, -0.15, -0.25,
                -0.35, -0.45, -0.55, -0.65, -0.75, -0.85, -0.95, -1.0, -1.1, -1.2, -1.3,
                -1.4, -1.5, -1.6, -1.7, -1.8, -1.9, -2.0, -2.1, -2.2, -2.3, -2.4, -2.5,
                -3.0, -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.05, 0.6,
                0.7, 0.8, 0.9, 0.15, 0.25, 0.35, 0.45, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
for Teff in [3500, 6000, 9000]:
    for logg in [3.0, 4.0, 5.0]:
        delta_I_mu_M_H.append(l1_losses(M_Hs, [Teff], [logg]))


# Convert to unit sensitivity.
unit_delta_sensitivity_M_H = np.mean(delta_I_mu_M_H, axis=0) / np.diff(M_Hs)
unit_delta_sensitivity_Teff = np.mean(delta_I_mu_Teff, axis=0) / np.diff(Teffs)
unit_delta_sensitivity_logg = np.mean(delta_I_mu_logg, axis=0) / np.diff(loggs)

# plt.scatter(np.diff(loggs), np.mean(delta_I_mu_logg, axis=0), s=10)
# plt.scatter(np.ones_like(np.diff(loggs)), unit_delta_sensitivity_logg, s=10)
# plt.show()

print("\nDelta 1 M_H [dex] --> Delta I(mu) = {}".format(np.mean(unit_delta_sensitivity_M_H)))
print("Delta 1 Teff [K] --> Delta I(mu) = {}".format(np.mean(unit_delta_sensitivity_Teff)))
print("Delta 1 logg [dex] --> Delta I(mu) = {}".format(np.mean(unit_delta_sensitivity_logg)))

# Standardize scaling, such that M_H rescaled has a radius of similarity = 1.
anchor_point = np.mean(unit_delta_sensitivity_M_H)
print("\nRescaled radius of similarity r(Delta M_H [dex]) = {}".format(anchor_point / np.mean(unit_delta_sensitivity_M_H)))
print("Rescaled radius of similarity r(Delta Teff [K]) = {} K".format(anchor_point / np.mean(unit_delta_sensitivity_Teff)))
print("Rescaled radius of similarity r(Delta logg [dex]) = {}".format(anchor_point / np.mean(unit_delta_sensitivity_logg)))
