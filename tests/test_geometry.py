from pytest import fixture
import dolfin as df
import numpy as np
from geometry import *


def test_init(mesh, markers, markerfunctions):
    geo = Geometry2D(mesh, markers=markers, markerfunctions=markerfunctions)
    assert geo.mesh == mesh
    assert geo.markers == markers
    assert geo.markerfunctions == markerfunctions
    assert geo.microstructure.f0 == None
    assert geo.microstructure.s0 == None
    assert geo.microstructure.n0 == None
    assert geo.crl_basis.c0 == None
    assert geo.crl_basis.r0 == None
    assert geo.crl_basis.l0 == None


def test_from_file(tmpdir):
    mesh = df.UnitSquareMesh(2,2)
    geo = Geometry2D(mesh)
    h5name = str(tmpdir+"geometry2d")
    geo.save(h5name)
    geo_load = Geometry2D.from_file(h5name+".h5")
    assert (geo_load.mesh.coordinates() == geo.mesh.coordinates()).all()
    assert geo_load.markerfunctions.vfun == None
    assert geo_load.markerfunctions.ffun == None
    assert geo_load.markerfunctions.cfun == None
    assert "efun" not in geo_load.markerfunctions._fields
    mesh = df.UnitCubeMesh(2,2,2)
    geo = Geometry(mesh)
    h5name = str(tmpdir+"geometry3d")
    geo.save(h5name)
    geo_load = Geometry.from_file(h5name+".h5")
    assert (geo_load.mesh.coordinates() == geo.mesh.coordinates()).all()
    assert geo_load.markerfunctions.vfun == None
    assert geo_load.markerfunctions.ffun == None
    assert geo_load.markerfunctions.cfun == None
    assert geo_load.markerfunctions.efun == None


def test_copy(mesh):
    geo = Geometry2D(mesh)
    geo_new = geo.copy(deepcopy=True)
    assert np.equal(geo.mesh.coordinates(), geo_new.mesh.coordinates()).all()
    assert geo.markers == geo_new.markers
    assert geo.markerfunctions.vfun == geo_new.markerfunctions.vfun
    assert geo.markerfunctions.ffun == geo_new.markerfunctions.ffun
    assert geo.markerfunctions.cfun == geo_new.markerfunctions.cfun
    geo = Geometry(mesh)
    geo_new = geo.copy(deepcopy=True)
    assert np.equal(geo.mesh.coordinates(), geo_new.mesh.coordinates()).all()
    assert geo.markers == geo_new.markers
    assert geo.markerfunctions.vfun == geo_new.markerfunctions.vfun
    assert geo.markerfunctions.efun == geo_new.markerfunctions.efun
    assert geo.markerfunctions.ffun == geo_new.markerfunctions.ffun
    assert geo.markerfunctions.cfun == geo_new.markerfunctions.cfun
    geo = HeartGeometry.from_file(example_meshes['ellipsoid'])
    geo_new = geo.copy(deepcopy=True)
    assert np.equal(geo.mesh.coordinates(), geo_new.mesh.coordinates()).all()
    assert geo.markers == geo_new.markers
    assert geo.markerfunctions.vfun == geo_new.markerfunctions.vfun
    assert geo.markerfunctions.efun == geo_new.markerfunctions.efun
    assert np.equal(geo.markerfunctions.ffun.array(),
                                geo_new.markerfunctions.ffun.array()).all()
    assert np.equal(geo.markerfunctions.cfun.array(),
                                geo_new.markerfunctions.cfun.array()).all()


def test_topology(mesh):
    geo = Geometry2D(mesh)
    assert geo.topology() == mesh.topology()


def test_dim(mesh):
    geo = Geometry2D(mesh)
    assert geo.dim() == mesh.topology().dim()


@fixture
def mesh():
    return df.UnitSquareMesh(2,2)

@fixture
def markers():
    return dict(top_marker=(10, 1), sides_marker=(30, 1), bottom_marker=(40, 1),
                lefthalf_marker=(50, 2))

@fixture
def markerfunctions(mesh, markers):
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

    return MarkerFunctions2D(cfun=cfun, ffun=ffun)
