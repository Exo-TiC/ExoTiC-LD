Quick start
===========

After installing the code (see :doc:`installation <installation>`) you are ready
to calculate limb-darkening coefficients. Below we demonstrate a minimal example.

First, we define the stellar parameters and which stellar models to use
in the computation.

.. code-block:: python

    # Path to store stellar and instrument data.
    ld_data_path = 'exotic_ld_data'

    # Stellar models grid.
    ld_model = 'mps1'

    # Metallicty [dex].
    M_H = 0.01

    # Effective temperature [K].
    Teff = 5512

    # Surface gravity [dex].
    logg = 4.47

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

    u1, u2 = sld.compute_quadratic_ld_coeffs(wavelength_range, mode)

Note that only the stellar and instrument data required for your calculation are
automatically downloaded. These data are saved to your specified ld_data_path, and
so on subsequent runs with the same parameters the calculation will run much faster.

The limb-darkening laws available are linear, quadratic, the Kipping
reparameterisation, square root, 3-parameter and 4-parameter non-linear. The
available stellar grids are listed in
:doc:`supported stellar grids <supported_stellar_grids>`,
and the available instrument modes are listed in
:doc:`supported instruments <supported_instruments>`.
