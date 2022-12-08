Installation
============

There are two steps for installing ExoTiC-LD.

 1) install the package with pip:

    ::

       pip install exotic-ld

 2) Download the stellar models and instrument throughputs from `this
    zenodo link <https://zenodo.org/record/6344946#.YistRy-l2ik>`_. Make sure
    that you download the version that matches the exotic-ld package version.

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
