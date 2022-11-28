Supported stellar grids
=======================

Here we list each of the available stellar grids.

Kurucz
------

    | ld_model = 'kurucz' or '1D'

The `Kurucz (1993) <https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/k93models/AA_README>`_
grid (CD-ROM No. 13) span a range of metallicities from -5.0 to 1.0,
effective temperatures from 3500 to 6500 k, and logg from 4.0 to 5.0.
There are 741 models in total, each evaluated at 1221 wavelengths and
17 radial positions on the stellar disc.

Stagger
-------

    | ld_model = 'stagger' or '3D'

The Stagger grid uses the `Magic (2015) <https://www.aanda.org/articles/aa/pdf/2015/01/aa23804-14.pdf>`_
3D stellar models and spans a range of metallicities from -3.0 to 0.0,
effective temperatures from 4000 to 7000 k, and logg from 1.5 to 5.0.
Note that this grid has non-uniform coverage, see figure 1 of
`Magic (2013) <https://www.aanda.org/articles/aa/pdf/2013/09/aa21274-13.pdf>`_.
There are 99 models in total, each evaluated at 105767 wavelengths and
10 radial positions on the stellar disc.

MPS-ATLAS-1
-----------

    | ld_model = 'mps1'

The MPS-ATLAS (set 1) grid uses the `Kostogryz (2022) <https://arxiv.org/pdf/2206.06641.pdf>`_
stellar models and spans a range of metallicities from -5.0 to 1.5,
effective temperatures from 3500 to 9000 k, and logg from 3.0 to 5.0.
This grid has extensive parameter space coverage, showcasing 34160 models
in total, each evaluated at 1221 wavelengths and 24 radial positions
on the stellar disc. Set 1 uses the Grevesse & Saul 1998 abundances
with a constant chemical mixing length of 1.25.

MPS-ATLAS-2
-----------

    | ld_model = 'mps2'

The MPS-ATLAS (set 2) grid uses the `Kostogryz (2022) <https://arxiv.org/pdf/2206.06641.pdf>`_
stellar models and spans a range of metallicities from -5.0 to 1.5,
effective temperatures from 3500 to 9000 k, and logg from 3.0 to 5.0.
This grid has extensive parameter space coverage, showcasing 34160 models
in total, each evaluated at 1221 wavelengths and 24 radial positions
on the stellar disc. Set 2 uses the the Asplund (2009) abundances with
a chemical mixing length dependent on stellar parameters from Viani (2018).

