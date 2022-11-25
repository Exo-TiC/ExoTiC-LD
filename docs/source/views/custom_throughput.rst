Custom throughput
=================

In this tutorial we take a quick look at how to use a custom throughput if
you cannot find the mode you are looking for in supported instruments.
Let us mock up some throughput data.

.. code-block:: python

    import numpy as np


    custom_wavelengths = np.linspace(10000., 20000., 100)
    custom_throughput = np.exp(-0.5 * ((custom_wavelengths - 15000.) / 5000.)**2)

With this data in hand, we can simply run the code as follows.

.. code-block:: python

    from exotic_ld import StellarLimbDarkening


    sld = StellarLimbDarkening(M_H=0.01, Teff=5512, logg=4.47, ld_model='1D',
                               ld_data_path='path/to/ExoTiC-LD_data')

    wavelength_range = [13000., 17000.]
    mode = 'custom'

    c1, c2, c3, c4 = sld.compute_4_parameter_non_linear_ld_coeffs(
        wavelength_range, mode, custom_wavelengths, custom_throughput)