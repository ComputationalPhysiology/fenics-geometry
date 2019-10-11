from pytest import fixture

from collections import namedtuple

import dolfin as df

from geometry import *
from geometry.utils import *


def test_set_namedtuple_default():
    NamedTuple1 = namedtuple('NamedTuple1', ['val1'])
    set_namedtuple_default(NamedTuple1)
    tup = NamedTuple1()
    assert tup.val1 is None
    NamedTuple2 = namedtuple('NamedTuple2', ['val1', 'val2'])
    set_namedtuple_default(NamedTuple2)
    tup = NamedTuple2()
    assert tup.val1 is None
    assert tup.val2 is None
    default = 1
    set_namedtuple_default(NamedTuple2, default)
    tup = NamedTuple2()
    assert tup.val1 == default
    assert tup.val2 == default


def test_load_geometry_from_h5(h5name):
    geo = load_geometry_from_h5(h5name)
    assert hasattr(geo, 'f0')
    assert hasattr(geo, 's0')
    assert hasattr(geo, 'n0')
    assert hasattr(geo, 'r0')
    assert hasattr(geo, 'c0')
    assert hasattr(geo, 'l0')
    assert hasattr(geo, 'cfun')
    assert hasattr(geo, 'vfun')
    assert hasattr(geo, 'efun')
    assert hasattr(geo, 'ffun')


def test_check_h5group(h5name):
    h5group = ""
    ggroup = "{}/geometry".format(h5group)
    mgroup = "{}/mesh".format(ggroup)
    lgroup = "{}/local basis functions".format(h5group)
    fgroup = "{}/microstructure/".format(h5group)
    assert check_h5group(h5name, ggroup) == True
    assert check_h5group(h5name, mgroup) == True
    assert check_h5group(h5name, lgroup) == False
    assert check_h5group(h5name, fgroup) == True
    assert check_h5group(h5name, "x") == False


def test_open_h5py(h5name):
    from h5py import File
    h5file = open_h5py(h5name)
    assert isinstance(h5file, File)


def test_load_local_basis(h5file):
    pass


@fixture
def h5name():
    return './demo/ellipsoid_20.h5'

@fixture
def h5file(h5name, ggroup, mgroup):
    with HDF5File(mpi_comm_world, h5name, "r") as h5file:
        # Load mesh
        mesh = Mesh(mpi_comm_world)
        h5file.read(mesh, mgroup, True)
    return h5file

@fixture
def ggroup(h5group=""):
    return "{}/geometry".format(h5group)

@fixture
def mgroup(ggroup):
    return "{}/mesh".format(ggroup)
