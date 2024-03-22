Installation
============

There is only one step (as of v3.2.0) for installing ExoTiC-LD.

 1) Using **python>=3.8**, install the package with pip:

    ::

       pip install exotic-ld

And now you are ready to go. Stellar models and instrument throughput data
are selected and downloaded automatically at runtime, *or custom files can
be input by the user*. These data are saved locally and so are only downloaded
once, speeding up subsequent runs. Head straight to the :doc:`quick start <quick_start>`
page to begin computing limb-darkening coefficients 🚀.

**Backwards compatibility/optional step:**

Prior to v3.2.0, the stellar models and instrument data had to be downloaded
manually. If you wish to not rely on an internet connection or you already
have these data, then you can proceed with step 2, which remains supported.
However this is no longer necessary, you can skip this step and start computing
limb-darkening coefficients right away.

 2) Download the stellar models and instrument throughputs from
    `this zenodo link <https://doi.org/10.5281/zenodo.7874921>`_.

    The downloaded and unzipped directory structure should have the following
    layout:

    ::

        exotic_ld_data/
        ├── Sensitivity_files/
        ├── kurucz/
        ├── stagger/
        ├── mps1/
        ├── mps2/

    Some of the stellar grids are very large in size, and so you can choose to
    download only the stellar grids that you require. The instrument throughputs
    directory, "Sensitivity_files", must always be included.

    You can place the downloaded directory anywhere on your machine, you'll
    just need to pass the path, "path/to/exotic_ld_data", as an input when
    running the code.
