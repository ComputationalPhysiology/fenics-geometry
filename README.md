[![Build Status](https://travis-ci.org/ComputationalPhysiology/fenics-geometry.svg?branch=master)](https://travis-ci.org/ComputationalPhysiology/fenics-geometry) [![Documentation Status](https://readthedocs.org/projects/fenics-geometry/badge/?version=latest)](https://fenics-geometry.readthedocs.io/en/latest/?badge=latest)


# README

A library handling geometries for Fenics-based problems. Based geometry.py as part of [pulse](https://github.com/ComputationalPhysiology/pulse) by Henrik Finsberg, and adapted by Alexandra Diem to fit arbitrary geometries.


## Overview

`fenics-geometry` provides a conventient way to create meshes for [FEniCS](https://fenicsproject.org), including cardiac meshes.


## Installation instructions


### Install with pip

<!--
`fenics-geometry` can be installed directly from [PyPI](https://pypi.org/project/fenics-geometry/)
```
pip install fenics-geometry
```
-->
The development version of `fenics-geometry` can directly be installed from git using pip
```
pip install git+https://github.com/KVSLab/fenics-geometry.git
```

<!--
### Install with conda
You can also install the package using `conda`
```
conda install -c conda-forge fenics-geometry
```
-->

<!--
### Docker
It is also possible to use Docker. There is a prebuilt docker image
using FEniCS 2019.1.0, Python3.6 and `fenics-geometry`. You can get it by typing
```
docker pull akdiem/fenics-geometry:latest
```
-->


### Requirements

* FEniCS version 2019.1.0 or newer

Note that if you install FEniCS using anaconda then you will not get support for parallel HDF5
see e.g [this issue](https://github.com/conda-forge/hdf5-feedstock/issues/51). We recommend installing FEniCS using [Docker](https://fenicsproject.org/download/)


## Getting started

Check out the demos in the demo folder.


## Automated test

Test are provided in the folder [`tests`](tests). You can run the test
with `pytest`
```
pytest -xv tests/
```
The tests are automatically run on [TravisCI](https://travis-ci.org/ComputationalPhysiology/fenics-geometry).


## Documentation

The documentation is built using ReadTheDocs and can be found at [https://fenics-geometry.readthedocs.io](https://fenics-geometry.readthedocs.io).


## Known issues

* If you encounter errors with `h5py` it needs to be built from scratch instead of installed from the binaries in `pip`:
```
pip uninstall h5py
pip install h5py --no-binary=h5py
```
* If you installed FEniCS using `conda` and encouter a `Fatal Python error: Aborted` with `<frozen importlib._bootstrap>` messages you need to specify the build for FEniCS during installation for `h5py` to work:
```
conda install fenics=2019.1.0=py37_5
```
