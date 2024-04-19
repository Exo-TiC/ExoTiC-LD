---
title: 'ExoTiC-LD: thirty seconds to stellar limb-darkening coefficients'
tags:
  - Python
  - astronomy
  - stellar models
  - limb darkening
  - transiting exoplanets
authors:
  - name: David Grant
    orcid: 0000-0001-5878-618X
    affiliation: '1'
    corresponding: true
  - name: Hannah R. Wakeford
    orcid: 0000-0003-4328-3867
    affiliation: '1'
affiliations:
  - name: HH Wills Physics Laboratory, University of Bristol, Tyndall Avenue, Bristol, BS8 1TL, UK
    index: 1
date: 19 April 2024
bibliography: paper.bib
 
---

# Summary

Stellar limb darkening is the observed variation in brightness of a star between 
its centre and edge (or limb) when viewed in the plane of the sky. Stellar 
brightness is maximal at the centre and then decreases radially and monotonically 
towards the limb – hence the term “limb darkening”. This effect is crucial for 
finding and characterising planets beyond our Solar System, known as exoplanets, 
as these planets are often studied when crossing in front of their host stars. 
As such, limb darkening is directly linked to the exoplanet signals. Limb 
darkening is typically modelled by one of various functional forms, as outlined 
in @Claret:2000 and @Sing:2010, and the coefficients of these functions is what 
`ExoTiC-LD` is designed to compute. A wide variety of functional forms are 
supported, including those benchmarked by @Espinoza:2016 as well as 
reparameterisations suggested by @Kipping:2013.

# Statement of need

Stellar limb darkening depends on the type of star, the wavelengths of light 
being observed, and the sensitivity of the instrument/telescope performing the 
observation. Therefore, to compute limb-darkening coefficients requires a 
frustrating amount of “data admin”. In brief, one must search grids of stellar 
models to find a good match with the science target in metallicity, effective 
temperature, and surface gravity. Then retrieve the wavelength-dependent 
sensitivity of the employed instrument. Process all these data into the correct 
form, and then finally compute the limb-darkening coefficients.

In `ExoTiC-LD` we have done all of the heavy lifting for the user, making the 
process as fast and frictionless as possible. A user simply has to `pip install` 
the code and the relevant data will be automatically downloaded at runtime and 
the limb-darkening coefficients computed. In particular, all the stellar and 
instrument data has been pre-processed and homogenised. Additionally, the 
stellar model grids have been stored as tree structures, enabling the efficient 
search for good matches and helpful warnings to the user.

`ExoTiC-LD` thus far has predominantly been utilised in the study of exoplanet 
atmospheres, helping to facilitate the study of Jupiter-like 
[e.g., @Alderson:2023; @Grant:2023], Neptune-like [e.g., @Roy:2023; @Radica:2024], 
and Earth-like exoplanets [e.g., @Moran:2023; @Kirk:2024]. Notably, it has 
also been incorporated into the popular open-source JWST data reduction and 
analysis pipeline, called `Eureka!` [@Bell:2022].

Currently, the stellar models supported are phoenix [@Husser:2013], kurucz 
[@Kurucz:1993], stagger [@Magic:2015], and MPS-ATLAS [@Kostogryz:2022; @Kostogryz:2023]. 
There are also options to provide custom data if the user has their own stellar 
models or instrument data. Similar codes are available, such as `JKTLD` 
[@Southworth:2015], `LDTk` [@Parviainen:2015], `ExoTETHyS` [@Morello:2020], 
and `exoCTK` [@Bourque:2021].

# Acknowledgements

We acknowledge contributions from Natasha Batalha, Matthew Hill, and Iva Laginja, 
as well as testing by Taylor Bell, Lili Alderson, Daniel Valentine, Charlotte 
Fairman, and Katy Chubb. D.G. was funded by UK Research and Innovation (UKRI) 
under the UK government’s Horizon Europe funding guarantee as part of an ERC 
Starter Grant [grant number EP/Y006313/1].

# References