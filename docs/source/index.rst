ExoTiC-LD
=========

ExoTiC-LD is a python package for calculating stellar limb-darkening
coefficients for specific instruments, stars, and wavelength ranges.

Stellar intensity (:math:`I`) is modelled, as a function of radial position
on the stellar disc (:math:`\mu`), from pre-computed grids of models spanning
a range of metallicity, effective temperature, and surface gravity. These
intensities are combined with an instrument's throughput and integrated over
a specified wavelength range, resulting in a one-dimensional profile,
:math:`I(\mu)`. Limb-darkening coefficients are then calculated by fitting
one of the various functional forms, as outlined in `Claret (2000) <https://ui.adsabs.harvard.edu/abs/2000A%26A...363.1081C/abstract>`_
and `Sing (2010) <https://ui.adsabs.harvard.edu/abs/2010A%26A...510A..21S/abstract>`_,
to the modelled :math:`I(\mu)` profile.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   Installation <views/installation>
   Quick start <views/quick_start>
   Supported stellar grids <views/supported_stellar_grids>
   Supported instruments <views/supported_instruments>
   Tutorials <views/tutorials>
   API <views/api/api>
   Citation <views/citation>

Acknowledgements
----------------

The present version of ExoTiC-LD is built by David Grant and Hannah Wakeford.

The original IDL code was translated into python by Matthew Hill with
improvements by Iva Laginja. The git history associated with the original
implementation can be found in the `ExoTiC-ISM <https://github.com/Exo-TiC/ExoTiC-ISM>`_
package from which this is a spin-off repository. We also thank Natasha Batalha
for providing the JWST throughput information from their PandExo package and
to Lili Alderson for reviewing and testing.

If you make use of ExoTiC-LD in your research, see the :doc:`citation page <views/citation>`
for info on how to cite this package and the underlying stellar models.

You can find other software from the Exoplanet Timeseries Characterisation
(ExoTiC) ecosystem over on `GitHub <https://github.com/Exo-TiC>`_.

