""" This library is based on the module pulse.geometry by Henrik Finsberg. It
has been extended to account for arbitrary geometries.
"""

from collections import namedtuple

from dolfin import (MeshFunction)

from .utils import set_namedtuple_default


Markers = namedtuple('Marker', ['label', 'value', 'dim'])
set_namedtuple_default(Markers)


class Geometry(object):

    def __init__(self, mesh=None, markers=None):
        self.mesh = mesh or {}
        self.markers = markers or {}


    def add_mesh(self, physics, mesh, markers=None):
        try:
            self.mesh[physics] = mesh
            self.markers[physics] = markers
        except TypeError:
            msg = "Can only set up multi-mesh geometry if Geometry() has been instantiated empty."
            raise TypeError(msg)


class Geometry2D(Geometry):

    def __init__(self, *args, **kwargs):
        super(Geometry2D, self).__init__(*args, **kwargs)
