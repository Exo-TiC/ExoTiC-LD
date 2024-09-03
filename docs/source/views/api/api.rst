API
===

This page provides an index of all functions and classes in the ExoTiC-LD
package, organized by module. Click on the links for detailed documentation.
For a step-by-step guide, you may prefer to check out the :doc:`tutorials <../tutorials>`.


Primary Interface
-----------------

The primary class in ExoTiC-LD is StellarLimbDarkening. On instantiation, this
class handles the stellar models. Next, there are methods for computing the
limb-darkening coefficients utilising the aforementioned stellar models. The
coefficients returned by each method can be found by clicking into the list
of methods. Additionally, the functional forms and coefficients for each
limb-darkening law can be found in the :doc:`ld_laws <./ld_laws>` subpackage
below.

.. toctree::
    :maxdepth: 2

    stellar_limb_darkening

Subpackages
-----------

The available limb-darkening laws are listed here. These functions are utilised by
the corresponding StellarLimbDarkening methods shown above. Functional forms and
coefficients for each law can be inspected by clicking into the functions.

.. toctree::
    :maxdepth: 2

    ld_laws
