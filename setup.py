from setuptools import setup


setup(
    name='exotic-ld',
    version='1.0.0',
    author='Hannah R Wakeford',
    author_email='hannah.wakeford@bristol.ac.uk',
    url='https://github.com/Exo-TiC/ExoTiC-LD',
    license='MIT',
    packages=['exotic_ld'],
    description='ExoTiC limb-darkening calculator',
    long_description=open("README.md").read(),
    package_data={
        '': ['README.md', 'LICENSE']
    },
    install_requires=['numpy', 'pandas', 'astropy', 'scipy'],
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
