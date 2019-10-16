[![Build Status](https://travis-ci.org/KVSlab/fenics-geometry.svg?branch=master)](https://travis-ci.org/KVSlab/fenics-geometry)

# fenics-geometry

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

## Requirements

* FEniCS version 2019.1.0 or newer

Note that if you install FEniCS using anaconda then you will not get support for parallel HDF5
see e.g [this issue](https://github.com/conda-forge/hdf5-feedstock/issues/51). We recommend installing FEniCS using [Docker](https://fenicsproject.org/download/)

## Getting started

Check out the demos in the demo folder.

<!--
## Automated test

Test are provided in the folder [`tests`](tests). You can run the test
with `pytest`
```
python -m pytest tests -vv
```
-->

<!--
## Documentation

Documentation can be found at [kvslab.github.io/fenics-geometry](https://kvslab.github.io/fenics-geometry)
-->

## Known issues

* If you encounter errors with `h5py` it needs to be built from scratch instead of installed from the binaries in `pip`:
```
pip uninstall h5py
pip install h5py --no-binary=h5py
```
