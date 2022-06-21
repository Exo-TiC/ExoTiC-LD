ExoTiC-LD
=========

Exoplanet Timeseries Characterisation - Limb-Darkening. A python
limb-darkening package to calculate the coefficients for specific
instruments, stars, and wavelength ranges.

ExoTiC-LD calculates limb-darkening parameters using a range of
functional forms, as outlined in `Claret (2010) <https://ui.adsabs.harvard.edu/abs/2000A%26A...363.1081C/abstract>`_
and `Sing (2010) <https://ui.adsabs.harvard.edu/abs/2010A%26A...510A..21S/abstract>`_.
The calculation is computed using the best matching stellar model from
either the 1D Kurucz stellar models or 3D `Magic et al. (2015) <https://ui.adsabs.harvard.edu/abs/2015A&A...573A..90M/abstract>`_
stellar models.

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   Installation <views/installation>
   Quick start <views/quick_start>
   Supported instruments <views/supported_instruments>
   Custom throughput <views/custom_throughput>
   Citation <views/citation>

Attribution
-----------

ExoTiC-LD was built from the original IDL code adapted by Hannah Wakeford
and translated into python by Matthew Hill with improvements by Iva Laginja,
and later on by David Grant. Thanks to Natasha Batalha for providing the
Webb throughput information from their PandExo package and to Lili Alderson
for reviewing and testing.

The git history associated with the original implementation can be found in
the `ExoTiC-ISM <https://github.com/Exo-TiC/ExoTiC-ISM>`_ package from which
this is a spin-off repository.

You can find other software from the Exoplanet Timeseries Characterisation
(ExoTiC) ecosystem over on `GitHub <https://github.com/Exo-TiC>`_.

