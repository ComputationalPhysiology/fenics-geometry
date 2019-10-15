from pytest import fixture

from collections import namedtuple

import dolfin as df

from geometry import *
from geometry.utils import *


def test_namedtuple_as_dict(markerfunctions2d, markerfunctions3d):
    mf2d_dict = namedtuple_as_dict(markerfunctions2d)
    for i, key in enumerate(markerfunctions2d._fields):
        assert key in mf2d_dict
        assert mf2d_dict[key] == markerfunctions2d[i]
    mf3d_dict = namedtuple_as_dict(markerfunctions3d)
    for i, key in enumerate(markerfunctions3d._fields):
        assert key in mf3d_dict
        assert mf3d_dict[key] == markerfunctions3d[i]
    empty_namedtuple = MarkerFunctions()
    empty_dict = namedtuple_as_dict(empty_namedtuple)
    for i, key in enumerate(empty_namedtuple._fields):
        assert key in empty_dict
        assert empty_dict[key] is None


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
    assert hasattr(geo, 'ffun')
    assert geo.f0 is None
    assert geo.s0 is None
    assert geo.n0 is None
    assert geo.r0 is None
    assert geo.c0 is None
    assert geo.l0 is None
    assert geo.cfun is None
    assert geo.vfun is None
    assert isinstance(geo.ffun, df.cpp.mesh.MeshFunctionSizet)
    assert geo.mesh.topology().dim() == 2


def test_save_geometry_to_h5(tmpdir, geometry2d):
    geo = geometry2d
    h5name = tmpdir+"geometry2d"
    save_geometry_to_h5(geo.mesh, h5name,
                        markers=namedtuple_as_dict(geo.markers),
                        markerfunctions=namedtuple_as_dict(geo.markerfunctions))
    h5name += ".h5"
    gload = load_geometry_from_h5(str(h5name))
    assert type(gload.f0) == type(geo.f0)
    assert type(gload.s0) == type(geo.s0)
    assert type(gload.n0) == type(geo.n0)
    assert type(gload.r0) == type(geo.r0)
    assert type(gload.c0) == type(geo.c0)
    assert type(gload.l0) == type(geo.l0)
    assert type(gload.vfun) == type(geo.vfun)
    assert type(gload.ffun) == type(geo.ffun)
    assert type(gload.cfun) == type(geo.cfun)
    print(geo.markers.keys())
    print(gload.markers.keys())
    for key, (marker, dim) in geo.markers.items():
        assert key in gload.markers.keys()
        assert gload.markers[key] == (marker, dim)


def test_check_h5group(h5name):
    h5group = ""
    ggroup = "{}/geometry".format(h5group)
    mgroup = "{}/mesh".format(ggroup)
    lgroup = "{}/local basis functions".format(h5group)
    fgroup = "{}/microstructure/".format(h5group)
    assert check_h5group(h5name, ggroup) == True
    assert check_h5group(h5name, mgroup) == True
    assert check_h5group(h5name, lgroup) == False
    assert check_h5group(h5name, fgroup) == False
    assert check_h5group(h5name, "x") == False


def test_open_h5py(h5name):
    from h5py import File
    h5file = open_h5py(h5name)
    assert isinstance(h5file, File)
    h5file.close()


@fixture
def mesh2d():
    return df.UnitSquareMesh(2,2)

@fixture
def mesh3d():
    return df.UnitCubeMesh(2,2,2)

@fixture
def markerfunctions2d(mesh2d):
    mesh = mesh2d
    vfun = df.MeshFunction("size_t", mesh, 0)
    vfun.set_all(0)
    ffun = df.MeshFunction("size_t", mesh, 1)
    ffun.set_all(1)
    cfun = df.MeshFunction("size_t", mesh, 2)
    cfun.set_all(2)
    return MarkerFunctions(vfun=vfun, ffun=ffun, cfun=cfun)

@fixture
def markerfunctions2d_invalid(mesh2d):
    mesh = mesh2d
    vfun = df.MeshFunction("size_t", mesh, 0)
    vfun.set_all(0)
    efun = df.MeshFunction("size_t", mesh, 1)
    efun.set_all(1)
    ffun = df.MeshFunction("size_t", mesh, 1)
    ffun.set_all(2)
    cfun = df.MeshFunction("size_t", mesh, 2)
    cfun.set_all(3)
    return MarkerFunctions(vfun=vfun, efun=efun, ffun=ffun, cfun=cfun)

@fixture
def markerfunctions3d(mesh3d):
    mesh = mesh3d
    vfun = df.MeshFunction("size_t", mesh, 0)
    vfun.set_all(0)
    efun = df.MeshFunction("size_t", mesh, 1)
    efun.set_all(1)
    ffun = df.MeshFunction("size_t", mesh, 2)
    ffun.set_all(2)
    cfun = df.MeshFunction("size_t", mesh, 3)
    cfun.set_all(3)
    return MarkerFunctions(vfun=vfun, efun=efun, ffun=ffun, cfun=cfun)

@fixture
def geometry2d():
    mesh = df.UnitSquareMesh(2,2)
    markers = dict(top_marker=(10, 1),
                    sides_marker=(30, 1),
                    bottom_marker=(40, 1),
                    lefthalf_marker=(50, 2))

    class Top(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[1], 1.0)

    class Sides(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[0], 1.0) or df.near(x[0], 0.0)

    class Bottom(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[1], 0.0)

    class LeftHalf(df.SubDomain):
        def inside(self, x, on_boundary):
            return x[0] < 0.5

    top = Top()
    sides = Sides()
    bottom = Bottom()
    left_half = LeftHalf()

    cfun = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    cfun.set_all(0)
    left_half.mark(cfun, markers["lefthalf_marker"][0])

    ffun = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    ffun.set_all(0)
    top.mark(ffun, markers["top_marker"][0])
    sides.mark(ffun, markers["sides_marker"][0])
    bottom.mark(ffun, markers["bottom_marker"][0])

    markerfunctions = MarkerFunctions2D(cfun=cfun, ffun=ffun)

    return Geometry2D(mesh, markers=markers, markerfunctions=markerfunctions)

@fixture
def h5name():
    return './tests/lv_slice.h5'

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
