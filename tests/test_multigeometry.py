from pytest import raises
import dolfin as df
from geometry import *


def test_init():
    geo1 = Geometry2D(df.UnitSquareMesh(2,2))
    geo2 = Geometry2D(df.UnitSquareMesh(3,3))
    geo3 = Geometry(df.UnitCubeMesh(2,2,2))
    with raises(Exception):
        assert MultiGeometry(geometries=[geo1, geo2, geo3],
                                                        labels=['a', 'b', 'c'])
        assert MultiGeometry(geometries=[geo1, geo2, geo3],
                                                        labels=['a', 'a', 'c'])
        assert MultiGeometry(geometries=[geo1, geo2, geo3],
                                                        labels=['a', 'b'])
    geo = MultiGeometry(geometries=[geo1, geo2], labels=['a', 'b'])
    assert geo.geometries['a'] == geo1
    assert geo.geometries['b'] == geo2


def test_add_geometry():
    geo1 = Geometry2D(df.UnitSquareMesh(2,2))
    geo2 = Geometry2D(df.UnitSquareMesh(3,3))
    geo = MultiGeometry()
    geo.add_geometry(geo1, 'a')
    with raises(Exception):
        assert geo.add_geometry(geo2, 'a')
    geo.add_geometry(geo2, 'b')
    assert geo.geometries['b'] == geo2
