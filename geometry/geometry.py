""" This library is based on the module pulse.geometry by Henrik Finsberg. It
has been extended to account for arbitrary geometries.
"""

from collections import namedtuple

from dolfin import (MeshFunction)

from .utils import (make_logger, set_namedtuple_default,
                    load_geometry_from_h5)

logger = make_logger(__name__, dolfin.get_log_level())


Markers = namedtuple('Markers', ['label', 'value', 'dim'])
set_namedtuple_default(Markers)

MarkerFunctions = namedtuple("MarkerFunctions", ["vfun", "efun", "ffun", "cfun"])
set_namedtuple_default(MarkerFunctions)

Microstructure = namedtuple("Microstructure", ["f0", "s0", "n0"])
set_default_none(Microstructure)


def get_attribute(obj, key1, key2, default=None):
    f = getattr(obj, key1, None)
    if f is None:
        f = getattr(obj, key2, default)
    return f


class Geometry(object):

    def __init__(self, mesh=None, markers=None, markerfunctions=None,
                    microstructure=None):
        self.mesh = mesh or {}
        self.markers = markers or {}
        self.markerfunctions = markerfunctions or {}
        self.microstructure = microstructure or {}


    def add_mesh(self, label, mesh, markers=None):
        try:
            self.mesh[label] = mesh
            self.markers[label] = markers or Markers()
            self.markerfunctions[label] = markerfunctions or MarkerFunctions()
            self.microstructure[label] = microstructure or None
        except TypeError:
            msg = "Can only set up multi-mesh geometry if Geometry() has been instantiated empty."
            raise TypeError(msg)


    @classmethod
    def from_file(cls, h5name, h5group="", comm=None):
        comm = comm if comm is not None else mpi_comm_world()
        return cls(**cls.load_from_file(h5name, h5group, comm))


class Geometry2D(Geometry):

    def __init__(self, *args, **kwargs):
        super(Geometry2D, self).__init__(*args, **kwargs)


class HeartGeometry(Geometry):

    def __init__(self, *args, **kwargs):
        super(HeartGeometry, self).__init__(*args, **kwargs)


    @staticmethod
    def load_from_file(h5name, h5group, comm):

        logger.debug("Load geometry from file {}".format(h5name))

        geo = load_geometry_from_h5(h5name, h5group, include_sheets=True, comm=comm)

        f0 = get_attribute(geo, "f0", "fiber", None)
        s0 = get_attribute(geo, "s0", "sheet", None)
        n0 = get_attribute(geo, "n0", "sheet_normal", None)

        c0 = get_attribute(geo, "c0", "circumferential", None)
        r0 = get_attribute(geo, "r0", "radial", None)
        l0 = get_attribute(geo, "l0", "longitudinal", None)

        vfun = get_attribute(geo, "vfun", None)
        ffun = get_attribute(geo, "ffun", None)
        cfun = get_attribute(geo, "cfun", "sfun", None)

        kwargs = {
            "mesh": geo.mesh,
            "markers": geo.markers,
            "markerfunctions": MarkerFunctions(vfun=vfun, ffun=ffun, cfun=cfun),
            "microstructure": Microstructure(f0=f0, s0=s0, n0=n0),
            "crl_basis": CRLBasis(c0=c0, r0=r0, l0=l0),
        }

        return kwargs
