Supported instrument modes
==========================

Here we list each of the available instrument modes.

Spectroscopic
-------------

    | Hubble STIS gratings : 'HST_STIS_G430L', 'HST_STIS_G750L'

    | Hubble WFC3 grisms   : 'HST_WFC3_G280p1', 'HST_WFC3_G280n1', 'HST_WFC3_G102',
    |                        'HST_WFC3_G141'

    | JWST NIRSpec : 'JWST_NIRSpec_Prism', 'JWST_NIRSpec_G395H', 'JWST_NIRSpec_G395M',
    |                'JWST_NIRSpec_G235H', 'JWST_NIRSpec_G235M', 'JWST_NIRSpec_G140H',
    |                'JWST_NIRSpec_G140M-f100', 'JWST_NIRSpec_G140H-f070', 'JWST_NIRSpec_G140M-f070'

    | JWST NIRISS  : 'JWST_NIRISS_SOSSo1', 'JWST_NIRISS_SOSSo2'

    | JWST NIRCam  : 'JWST_NIRCam_F322W2', 'JWST_NIRCam_F444'

    | JWST MIRI    : 'JWST_MIRI_LRS'

Note for the WFC3 G280 grism the p1 and n1 signify the positive 1st order
spectrum and negative 1st order spectrum for the UVIS grism.

.. figure:: images/Supported_spectroscopic_modes.png
   :alt: supported spectroscopic modes

Photometric
-----------

    | Spitzer IRAC : 'Spitzer_IRAC_Ch1', 'Spitzer_IRAC_Ch2'

    | TESS : 'TESS'

Note for Spitzer Ch1 is the 3.6 micron channel and Ch2 is the 4.5 microns channel.

.. figure:: images/Supported_photometric_modes.png
   :alt: supported photometric modes

Custom
------

    | Custom profile : 'custom'

See the custom throughput :doc:`tutorial <tutorials/custom_throughput>`.
