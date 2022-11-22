Quick start
===========

After installing the code and downloading the accompanying data (see
installation) you are ready to compute limb-darkening coefficients. Below
we demonstrate a minimal example. First we define the stellar parameters
and which stellar models to use in the computation.

.. code-block:: python

    # Metallicty [dex].
    M_H = 0.01

    # Effective temperature [K].
    Teff = 5512

    # Gravity [dex].
    logg = 4.47

    # Stellar models: 1D or 3D grid.
    ld_model = '1D'

    # Path to the installed data.
    ld_data_path = 'path/to/ExoTiC-LD_data'

Next, import the StellarLimbDarkening class and set the stellar parameters.

.. code-block:: python

    from exotic_ld import StellarLimbDarkening


    sld = StellarLimbDarkening(M_H, Teff, logg, ld_model, ld_data_path)

Now you can compute the stellar limb-darkening coefficients for the
limb-darkening law of your choice. You simply have to specify the instrument
mode and the wavelength range you require.

.. code-block:: python

    # Start and end of wavelength interval [angstroms].
    wavelength_range = [20000., 30000.]

    # Instrument mode.
    mode = 'JWST_NIRSpec_prism'

    c1, c2 = sld.compute_quadratic_ld_coeffs(wavelength_range, mode)

The limb-darkening laws available are linear, quadratic, 3-parameter and
4-parameter non-linear. The available instrument modes are listed in the
supported instruments page.
