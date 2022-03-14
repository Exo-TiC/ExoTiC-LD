# ExoTiC-LD

**Exoplanet Timeseries Characterisation - Limb-Darkening**

Limb-darkening package to calculate the coefficients for specific instruments, stars, and wavelength ranges.

This code calculates limb-darkening parameters using a range of functional forms, as outlined in [Claret (2010)](https://ui.adsabs.harvard.edu/abs/2000A%26A...363.1081C/abstract) and [Sing (2010)](https://ui.adsabs.harvard.edu/abs/2010A%26A...510A..21S/abstract).
This calculation is computed using 1D Kurucz stellar models or 3D stellar models for a smaller subset of parameters from [Magic et al. (2015)](https://ui.adsabs.harvard.edu/abs/2015A&A...573A..90M/abstract).

This package was built from the original IDL code adapted by Hannah Wakeford and translated into python by Matthew Hill with improvements by Iva Laginja. The git history associated with these steps can be found in the [ExoTiC-ISM](https://github.com/Exo-TiC/ExoTiC-ISM) package from which this is a spin-off repository.

## Install

This package is installable via pip using the following command

         pip install exotic-ld

Alternatively you can clone this repository and use this as a standard python script. 

		git clone https://github.com/Exo-TiC/ExoTiC-LD.git

or

		git clone git@github.com:Exo-TiC/ExoTiC-LD.git

Using both 'pip' and 'git' you will need to download the required data detailed below.

## Data
To run this package you will need to download the stellar model grids and supported instrument information from [ExoTiC-LD_data on zenodo](https://zenodo.org/record/6344946#.YistRy-l2ik).

The location saved locally is then used as an input to the function.


## How to Run the code

         from exotic_ld import limb_dark_fit
         import numpy as np

         # Set the observing mode - see below for available doc strings
         mode = 'WFC3_G141'
         
         # Give it the wavelength range you want the limb-darkening to be calculated over
         # This can be from the data itself, or a pregenerated range like below, but must be a numpy array
         wsdata = np.arange(11100,11200,0.5) 

         # Set up the stellar parameters
         M_H = 0.1
         Teff = 6545
         logg = 4.2

         # Tell it where the data from the Zenodo link has been placed
         dirsen = '/Users/iz19726/Downloads/LD_data/' 
         
         # Tell it which stellar model grid you would like to use: '1D' or '3D'
         ld_model = '1D' 

         result = limb_dark_fit(mode, wsdata, M_H, Teff, logg, dirsen, ld_model='1D')

The returned result contains the coefficients for all versions of the limb-darkening equation considered
	 
	 uLD, c1, c2, c3, c4, cp1, cp2, cp3, cp4, aLD, bLD

where:
- uLD: float; linear limb darkening coefficient
- aLD, bLD: float; quadratic limb darkening coefficients
- cp1, cp2, cp3, cp4: float; three-parameter limb darkening coefficients
- c1, c2, c3, c4: float; non-linear limb-darkening coefficients

**NOTE:** There is a current issue open to add a selection criteria so that only the desired coefficients are returned. 

## Supported telescope and instrument modes
Supported instrument mode doc strings:

### Spectroscopic:
**Hubble** *STIS gratings*: 'STIS_G750L', 'STIS_G430L' 

**Hubble** *WFC3 grisms*: 'WFC3_G280p1', 'WFC3_G280n1', 'WFC3_G102', 'WFC3_G141'

Note for the WFC3 G280 grism the p1 and n1 signify the positive 1st order spectrum and negative 1st order spectrum for the UVIS grism. 

**Webb** *NIRSpec*: 'NIRSpec_Prism', 'NIRSpec_G395H', 'NIRSpec_G395M', 'NIRSpec_G235H', 'NIRSpec_G235M', 'NIRSpec_G140Hf100', 'NIRSpec_G140Mf100', 'NIRSpec_G140Hf070', 'NIRSpec_G140Mf070'

**Webb** *NIRISS*: 'NIRISS_SOSSo1', 'NIRISS_SOSSo2'

**Webb** *NIRCam*: 'NIRCam_F322W2', 'NIRCam_F444'

**Webb** *MIRI*: 'MIRI_LRS'

### Photometric:

**TESS**: 'TESS'

**Spitzer** *IRAC*: 'IRAC_Ch1', 'IRAC_Ch2'

Where Ch1 (3.6 microns), Ch2 (4.5 microns)

**NOTE:** There is a current issue open to also allow for custom instrument throughputs to be implemented. 

<img src="Supported_spectroscopic_modes.png" width="80%" />  
<img src="Supported_photometric_modes.png" width="80%" />  


## About this repository

### Contributing and code of conduct

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines, and the process for submitting issues and pull requests to us.
Please also see our [CODE OF CONDUCT](CODE_OF_CONDUCT.md).

If you use this code in your work, please find citation snippets to give us credits with in [CITATION.txt](CITATION.txt).

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.txt) file for details.

### Acknowledgments

* Hannah Wakeford for IDL code and adaption to an independent Python package - [@hrwakeford](https://github.com/hrwakeford)
* Matthew Hill for a functional translation from IDL to Python - [@mattjhill](https://github.com/mattjhill)
* Iva Laginja for implementing improvements to the script - [@ivalaginja](https://github.com/ivalaginja)
* Natasha Batalha for providing the Webb throughput information from their PandExo package - [@natashabatalha](https://github.com/natashabatalha)
* David Grant for making it pip installable - [@davogrant](https://github.com/DavoGrant)
* Lili Alderson for reviewing and testing - [@lili-alderson](https://github.com/lili-alderson)
