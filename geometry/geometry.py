""" This library is based on the module pulse.geometry by Henrik Finsberg. It
has been extended to account for arbitrary geometries.
"""

from collections import namedtuple

from dolfin import (MeshFunction, LogLevel)
import dolfin as df

from .utils import (set_namedtuple_default, namedtuple_as_dict,
                    mpi_comm_world, load_geometry_from_h5,
                    save_geometry_to_h5)


Markers = namedtuple('Markers', ['label', 'value', 'dim'])
set_namedtuple_default(Markers)

MarkerFunctions = namedtuple("MarkerFunctions", ["vfun", "efun", "ffun", "cfun"])
set_namedtuple_default(MarkerFunctions)

MarkerFunctions2D = namedtuple("MarkerFunctions2D", ["vfun", "ffun"])
set_namedtuple_default(MarkerFunctions2D)

Microstructure = namedtuple("Microstructure", ["f0", "s0", "n0"])
set_namedtuple_default(Microstructure)

CRLBasis = namedtuple("CRLBasis", ["c0", "r0", "l0"])
set_namedtuple_default(CRLBasis)


def get_attribute(obj, key1, key2, default=None):
    f = getattr(obj, key1, None)
    if f is None and key2 is not None:
        f = getattr(obj, key2, default)
    return f


class Geometry(object):

    def __init__(self, mesh=None, markers=None, markerfunctions=None,
                    microstructure=None, crl_basis=None):
        self.mesh = mesh or {}
        self.markers = markers or {}
        self.markerfunctions = markerfunctions or {}
        self.microstructure = microstructure or {}
        self.crl_basis = crl_basis or {}


    @classmethod
    def from_file(cls, h5name, h5group="", comm=None):
        comm = comm if comm is not None else mpi_comm_world
        return cls(**cls.load_from_file(h5name, h5group, comm))


    @staticmethod
    def load_from_file(h5name, h5group, comm):

        df.begin(LogLevel.PROGRESS, "Load mesh from h5 file")
        geo = load_geometry_from_h5(h5name, h5group, include_sheets=False,
                                    comm=comm)
        df.end()

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


    def save(self, h5name, h5group="", other_functions=None,
                other_attributes=None, overwrite_file=False,
                overwrite_group=True):
        df.begin(LogLevel.PROGRESS, "Saving geometry to {}...".format(h5name))
        save_geometry_to_h5(self.mesh, h5name, h5group=h5group,
                    markers=self.markers,
                    markerfunctions=namedtuple_as_dict(self.markerfunctions),
                    microstructure=namedtuple_as_dict(self.microstructure),
                    local_basis=namedtuple_as_dict(self.crl_basis),
                    overwrite_file=overwrite_file,
                    overwrite_group=overwrite_group)
        df.end()


    def topology(self):
        return self.mesh.topology()


    def dim(self):
        return self.mesh.geometry().dim()


    @property
    def dx(self):
        return df.dx(domain=self.mesh, subdomain_data=self.cfun)

    @property
    def ds(self):
        return df.ds(domain=self.mesh, subdomain_data=self.ffun)

    @property
    def vfun(self):
        return self.markerfunctions.vfun

    @property
    def efun(self):
        return self.markerfunctions.efun

    @property
    def ffun(self):
        return self.markerfunctions.ffun

    @property
    def cfun(self):
        return self.markerfunctions.cfun

    @property
    def f0(self):
        return self.microstructure.f0

    @property
    def s0(self):
        return self.microstructure.s0

    @property
    def n0(self):
        return self.microstructure.n0

    @property
    def c0(self):
        return self.crl_basis.c0

    @property
    def l0(self):
        return self.crl_basis.c0

    @property
    def r0(self):
        return self.crl_basis.c0



class MixedGeometry(object):

    def __init__(self, geometries=None, labels=None):
        self.geometries = {}
        for geo, l in zip(geometries, labels):
            self.geometries[l] = geo


    def add_geometry(self, geometry, label):
        if not isinstance(geometry, Geometry):
            msg = "Can only add instances of Geometry. You tried to add an instance of {}".format(type(geometry))
            raise TypeError(msg)
        self.geometries[label] = geometry


class Geometry2D(Geometry):

    def __init__(self, *args, **kwargs):
        super(Geometry2D, self).__init__(*args, **kwargs)


class HeartGeometry(Geometry):

    def __init__(self, *args, **kwargs):
        super(HeartGeometry, self).__init__(*args, **kwargs)


    @staticmethod
    def load_from_file(h5name, h5group, comm):

        df.begin(LogLevel.PROGRESS, "Load mesh from h5 file")
        geo = load_geometry_from_h5(h5name, h5group, include_sheets=True,
                                    comm=comm)
        df.end()

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
