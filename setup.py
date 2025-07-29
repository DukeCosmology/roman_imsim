from setuptools import setup

setup(
    name="roman_imsim",
    version="2.0",
    description="Image simulations suite for the Roman Space Telescope",
    license="MIT",
    author="Michael Troxel",
    author_email="michael.troxel@duke.edu",
    url="https://github.com/DukeCosmology/roman_imsim",
    packages=["roman_imsim"],
    install_requires=["galsim", "fitsio", "astropy", "numpy"],
)
