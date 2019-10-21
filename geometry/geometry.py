""" This library is based on the module pulse.geometry by Henrik Finsberg. It
has been extended to account for arbitrary geometries.
"""

from collections import namedtuple

from dolfin import (MeshFunction, LogLevel)
import dolfin as df

from .utils import (set_namedtuple_default, namedtuple_as_dict,
                    mpi_comm_world, load_geometry_from_h5,
                    save_geometry_to_h5)

MarkerFunctions = namedtuple("MarkerFunctions", ["vfun", "efun", "ffun", "cfun"])
set_namedtuple_default(MarkerFunctions)

MarkerFunctions2D = namedtuple("MarkerFunctions2D", ["vfun", "ffun", "cfun"])
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
        """A Geometry object for FEniCS applications.

        The geometry can either be instantiated directly from a mesh

        Example
        -------

            .. code-block:: python

                import dolfin
                mesh = dolfin.UnitCubeMesh(3,3,3)
                geo = Geometry(mesh)

        or it can be instantiated by loading a geometry from a file.

        Example
        -------

            .. code-block:: python

                # Geometry is stored in a file "geometry.h5"
                geo = Geometry.from_file("geometry.h5")

        Parameters
        ----------
        mesh : :class:`dolfin.mesh`
            The mesh
        markers : dict
            A dictionary containing mesh markers, where the key is the label and
            the value is a tuple of the marker and the topological dimension.
        markerfunctions : MarkerFunctions or MarkerFunctions2D
            A namedtuple containing marker functions. In the 3D case these are
            a vertex function (vfun), edge function (efun), facet function
            (ffun), and cell function (cfun). The 2D case does not contain
            efun.
        microstructure : Microstructure
            A namedtuple containing the microstructure fields. These are
            fibres (f0), sheets (s0), and sheet normals (n0).
        crl_basis : CRLBasis
            A namedtuple containing a CRL (circumferential, radial,
            longitudinal) basis.
        """
        self.mesh = mesh or {}
        self.markers = markers or {}
        self.markerfunctions = markerfunctions or MarkerFunctions()
        self.microstructure = microstructure or Microstructure()
        self.crl_basis = crl_basis or CRLBasis()


    @classmethod
    def from_file(cls, h5name, h5group="", comm=None):
        """Loads a geometry from a h5 file. The function has to be called from
        the subclass that should instantiate the geometry

        Example
        -------

            .. code-block:: python

                # A 2D geometry is stored in a file "geometry2d.h5"
                geo = Geometry2D.from_file("geometry2d.h5")

                # A heart geometry is stored in a file "heart_geometry.h5"
                geo = HeartGeometry.from_file("heart_geometry.h5")

        Parameters
        ----------
        cls : class
            The class from which the function is called.
        h5name : str
            The path to the h5 file containing the geometry.
        h5group : str
            The h5 group of the geometry.
        comm :
            MPI communicator.
        """
        comm = comm if comm is not None else mpi_comm_world
        return cls(**cls.load_from_file(h5name, h5group, comm))


    @classmethod
    def load_from_file(cls, h5name, h5group, comm):

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
        efun = get_attribute(geo, "efun", None)
        ffun = get_attribute(geo, "ffun", None)
        cfun = get_attribute(geo, "cfun", "sfun", None)

        kwargs = {
            "mesh": geo.mesh,
            "markers": geo.markers,
            "markerfunctions": MarkerFunctions(vfun=vfun, efun=efun, ffun=ffun,
                                                cfun=cfun),
            "microstructure": Microstructure(f0=f0, s0=s0, n0=n0),
            "crl_basis": CRLBasis(c0=c0, r0=r0, l0=l0),
        }

        return kwargs


    def save(self, h5name, h5group="", other_functions=None,
                other_attributes=None, overwrite_file=False,
                overwrite_group=True):
        """Saves a geometry to a h5 file.

        Parameters
        ----------
        h5name : str
            The location and name of the output file without file extension.
        h5group : str
            The h5 group of the geometry.
        other_functions :
            Extra mesh functions.
        other_attributes :
            Extra mesh attributes.
        overwrite_file : bool
            True if existing file should be overwritten.
        overwrite_group : bool
            True if existing group should be overwritten.
        """

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
        """Returns the topology of the geometry.
        """
        return self.mesh.topology()


    def dim(self):
        """Returns the topological dimension of the geometry.
        """
        return self.mesh.geometry().dim()


    @property
    def facet_normal(self):
        return df.FacetNormal(self.mesh)

    @property
    def dx(self):
        """Returns the mesh volume measure using `self.cfun` as `subdomain_data`
        """
        return df.dx(domain=self.mesh, subdomain_data=self.cfun)

    @property
    def ds(self):
        """Returns the mesh surface measure using `self.ffun` as `subdomain_data`
        """
        return df.ds(domain=self.mesh, subdomain_data=self.ffun)

    @property
    def vfun(self):
        """Vertex mesh function.
        """
        return self.markerfunctions.vfun

    @property
    def efun(self):
        """Edge mesh function. Not available if `self.dim() < 3`.
        """
        return self.markerfunctions.efun

    @property
    def ffun(self):
        """Facet mesh function.
        """
        return self.markerfunctions.ffun

    @property
    def cfun(self):
        """Cell mesh function.
        """
        return self.markerfunctions.cfun

    @property
    def f0(self):
        """Fibre microstructure.
        """
        return self.microstructure.f0

    @property
    def s0(self):
        """Sheet microstructure.
        """
        return self.microstructure.s0

    @property
    def n0(self):
        """Sheet normals microstructure.
        """
        return self.microstructure.n0

    @property
    def c0(self):
        """Circumferential basis.
        """
        return self.crl_basis.c0

    @property
    def l0(self):
        """Longitudinal basis.
        """
        return self.crl_basis.c0

    @property
    def r0(self):
        """Radial basis.
        """
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
        """A 2D Geometry object for FEniCS applications. Requires the use of
        `MarkerFunctions2D`. Refer to :class:`.Geometry` for more information.
        """

        if 'markerfunctions' in kwargs and\
                not isinstance(kwargs['markerfunctions'], MarkerFunctions2D):
            msg = "Marker functions is of type {}. Type {} is required.".format(
                    type(kwargs['markerfunctions']), MarkerFunctions2D
            )
            raise TypeError(msg)
        super(Geometry2D, self).__init__(*args, **kwargs)


    @classmethod
    def load_from_file(cls, h5name, h5group, comm):

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
            "markerfunctions": MarkerFunctions2D(vfun=vfun, ffun=ffun, cfun=cfun),
            "microstructure": Microstructure(f0=f0, s0=s0, n0=n0),
            "crl_basis": CRLBasis(c0=c0, r0=r0, l0=l0),
        }

        return kwargs


class HeartGeometry(Geometry):

    def __init__(self, *args, **kwargs):
        """A Heart Geometry object for FEniCS applications. Refer to
        :class:`.Geometry` for more information.
        """
        super(HeartGeometry, self).__init__(*args, **kwargs)
