[![Build](https://github.com/DukeCosmology/roman_imsim/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/DukeCosmology/roman_imsim/actions/workflows/build.yml)
[![End-to-end Test](https://github.com/DukeCosmology/roman_imsim/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DukeCosmology/roman_imsim/actions/workflows/ci.yml)

# roman_imsim


Nonstandard dependencies are represented in setup.py.

See wiki for information on how to run and documentation.

## installation

Installing this software does _not_ install certain data that the user may expect, such as stellar SED libraries.
Information about downloading those data can be found [here](https://lsstdesc.org/skyCatalogs/installation.html#install-needed-data-files).
Additionally, the relevant environment variables (i.e., `SIMS_SED_LIBRARY_DIR`) need be set for skyCatalogs to work properly.

## roman_imsim_testdata

Testing data is provided as a [git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) pointing to <https://github.com/DukeCosmology/roman_imsim_testdata>.

To access these data, run
```bash
git submodule init
git submodule update
```
