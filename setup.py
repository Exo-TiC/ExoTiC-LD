from setuptools import setup


setup(
    name='exotic-ld',
    version='3.2.0',
    author='David Grant and Hannah R Wakeford',
    author_email='hannah.wakeford@bristol.ac.uk',
    url='https://github.com/Exo-TiC/ExoTiC-LD',
    license='MIT',
    packages=['exotic_ld', 'grid_build.kd_trees'],
    description='ExoTiC limb-darkening calculator',
    long_description="Calculate limb-darkening coefficients for specific "
                     "instruments, stars, and wavelength ranges.",
    python_requires='>=3.8.0',
    install_requires=['scipy>=1.8.0', 'numpy', 'requests', 'tqdm'],
    package_data={
        'grid_build.kd_trees': ['*.pickle']
    },
    include_package_data=True,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    zip_safe=True,
)
