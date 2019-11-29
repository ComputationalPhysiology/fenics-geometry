""" This library is based on the module pulse.geometry by Henrik Finsberg. It
has been extended to account for arbitrary geometries.
"""

from collections import namedtuple

from dolfin import (LogLevel, Mesh, MeshFunction)
import dolfin as df

from .utils import (load_geometry_from_h5, map_vector_field, mpi_comm_world,
                    namedtuple_as_dict, save_geometry_to_h5,
                    set_namedtuple_default,)

MarkerFunctions_ = namedtuple("MarkerFunctions", ["vfun", "efun", "ffun", "cfun"])
set_namedtuple_default(MarkerFunctions_)

class MarkerFunctions(MarkerFunctions_):
    """A collection of mesh marker functions.

    Parameters
    ----------
    vfun : :class:`dolfin.cpp.mesh.MeshFunction`
        A mesh function for vertex markers.
    efun : :class:`dolfin.cpp.mesh.MeshFunction`
        A mesh function for edge markers.
    ffun : :class:`dolfin.cpp.mesh.MeshFunction`
        A mesh function for facet markers.
    cfun : :class:`dolfin.cpp.mesh.MeshFunction`
        A mesh function for cell markers.
    """
    pass

MarkerFunctions2D_ = namedtuple("MarkerFunctions2D", ["vfun", "ffun", "cfun"])
set_namedtuple_default(MarkerFunctions2D_)

class MarkerFunctions2D(MarkerFunctions2D_):
    """A collection of mesh marker functions for 2D geometries.

    Parameters
    ----------
    vfun : :class:`dolfin.cpp.mesh.MeshFunction`
        A mesh function for vertex markers.
    ffun : :class:`dolfin.cpp.mesh.MeshFunction`
        A mesh function for facet markers.
    cfun : :class:`dolfin.cpp.mesh.MeshFunction`
        A mesh function for cell markers.
    """
    pass

Microstructure_ = namedtuple("Microstructure", ["f0", "s0", "n0"])
set_namedtuple_default(Microstructure_)

class Microstructure(Microstructure_):
    """A collection of the microstructure fields of the geometry.

    Parameters
    ----------
    f0 : :class:`dolfin.function.function.Function`
        A function containing the fibre field.
    s0 : :class:`dolfin.function.function.Function`
        A function containing the sheet field.
    s0 : :class:`dolfin.function.function.Function`
        A function containing the sheet normal field.
    """
    pass

CRLBasis_ = namedtuple("CRLBasis", ["c0", "r0", "l0"])
set_namedtuple_default(CRLBasis_)

class CRLBasis(CRLBasis_):
    """A collection of local basis functions (circumferential, radial,
    longitudinal).

    Parameters
    ----------
    c0 : :class:`dolfin.function.function.Function`
        A function containing the circumferential basis.
    r0 : :class:`dolfin.function.function.Function`
        A function containing the radial basis.
    l0 : :class:`dolfin.function.function.Function`
        A function containing the sheet longitudinal basis.
    """
    pass


def get_attribute(obj, key1, key2=None, default=None):
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
        return cls(**cls._load_from_file(h5name, h5group, comm))


    @classmethod
    def _load_from_file(cls, h5name, h5group, comm):

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


    def copy(self, deepcopy=False):
        """Returns a copy of self.

        Parameters
        ----------
        deepcopy : bool
            True if copy should be deep, default False.
        """
        msg = "Copying geometry"
        if deepcopy:
            msg += " with deepcopy."
        df.begin(LogLevel.DEBUG, msg)
        cp = self.__class__(**self._copy(deepcopy))
        df.end()
        return cp


    def _copy(self, deepcopy):
        new_mesh = Mesh(self.mesh)

        new_markerfunctions = {}
        for dim, fun in ((0, "vfun"), (1, "efun"), (2, "ffun"), (3, "cfun")):
            f_old = get_attribute(self, fun)
            if f_old is None:
                continue
            f = MeshFunction("size_t", new_mesh, dim, new_mesh.domains())
            f.set_values(f_old.array())
            new_markerfunctions[fun] = f
        markerfunctions = MarkerFunctions(**new_markerfunctions)

        new_microstructure = {}
        for field in ("f0", "s0", "n0"):
            v0_old = get_attribute(self, field)
            if v0_old is None:
                continue
            v0 = map_vector_field(v0_old, new_mesh)
            new_microstructure[field] = v0
        microstructure = Microstructure(**new_microstructure)

        new_crl_basis = {}
        for basis in ("c0", "r0", "l0"):
            v0_old = get_attribute(self, basis)
            if v0_old is None:
                continue
            v0 = map_vector_field(v0_old, new_mesh)
            new_crl_basis[basis] = v0
        crl_basis = CRLBasis(**new_crl_basis)

        return dict(
            mesh = new_mesh,
            markers = self.markers,
            markerfunctions = markerfunctions,
            microstructure = microstructure,
            crl_basis = crl_basis,
        )


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
    def _load_from_file(cls, h5name, h5group, comm):

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


    def copy(self, deepcopy=False):
        return super(Geometry2D, self).copy(deepcopy=deepcopy)


    def _copy(self, deepcopy):
        new_mesh = Mesh(self.mesh)

        new_markerfunctions = {}
        for dim, fun in ((0, "vfun"), (1, "ffun"), (2, "cfun")):
            f_old = get_attribute(self, fun)
            if f_old is None:
                continue
            f = MeshFunction("size_t", new_mesh, dim, new_mesh.domains())
            f.set_values(f_old.array())
            new_markerfunctions[fun] = f
        markerfunctions = MarkerFunctions2D(**new_markerfunctions)

        new_microstructure = {}
        for field in ("f0", "s0", "n0"):
            v0_old = get_attribute(self, field)
            if v0_old is None:
                continue
            v0 = map_vector_field(v0_old, new_mesh)
            new_microstructure[field] = v0
        microstructure = Microstructure(**new_microstructure)

        new_crl_basis = {}
        for basis in ("c0", "r0", "l0"):
            v0_old = get_attribute(self, basis)
            if v0_old is None:
                continue
            v0 = map_vector_field(v0_old, new_mesh)
            new_crl_basis[basis] = v0
        crl_basis = CRLBasis(**new_crl_basis)

        return dict(
            mesh = new_mesh,
            markers = self.markers,
            markerfunctions = markerfunctions,
            microstructure = microstructure,
            crl_basis = crl_basis,
        )


class HeartGeometry(Geometry):

    def __init__(self, *args, **kwargs):
        """A Heart Geometry object for FEniCS applications. Refer to
        :class:`.Geometry` for more information.
        """
        super(HeartGeometry, self).__init__(*args, **kwargs)


    @classmethod
    def _load_from_file(cls, h5name, h5group, comm):
        return super()._load_from_file(h5name, h5group, comm)


    def get_lv_marker(self):
        if "ENDO" in self.geometry.markers:
            return self.geometry.markers["ENDO"][0]
        elif "ENDO_LV" in self.geometry.markers:
            return self.geometry.markers["ENDO_LV"][0]
        else:
            raise KeyError("Geometry is missing marker for ENDO_LV/ENDO.")


    def get_rv_marker(self):
        if not self.has_rv():
            raise KeyError("Geometry is not biventricular.")
        elif "ENDO_RV" in self.geometry.markers:
            return self.geometry.markers["ENDO_RV"][0]
        else:
            raise KeyError("Geometry is missing marker for ENDO_RV.")


    def has_rv(self):
        return "ENDO_RV" in self.markers.keys()


    def get_epi_marker(self):
        if "EPI" in self.geometry.markers:
            return self.geometry.markers["EPI"][0]
        else:
            raise KeyError("Geometry is missing marker for EPI.")


    def get_base_marker(self):
        if "BASE" in self.geometry.markers:
            return self.geometry.markers["BASE"][0]
        else:
            raise KeyError("Geometry is missing marker for BASE.")


    def copy(self, deepcopy=False):
        return super(HeartGeometry, self).copy(deepcopy=deepcopy)


    def _copy(self, deepcopy):
        return super(HeartGeometry, self)._copy(deepcopy)


class MultiGeometry(object):

    def __init__(self, geometries=None, labels=None):
        """An object for multiple geometries for FEniCS applications.

        The multi-geometry can be instantiated directly from a collection
        of Geometry objects

        Example
        -------

            .. code-block:: python

                import dolfin
                mesh1 = dolfin.UnitSquareMesh(3,3)
                mesh2 = dolfin.UnitSquareMesh(5,5)
                geo1 = Geometry2D(mesh1)
                geo2 = Geometry2D(mesh2)
                geo = MutliGeometry(geometries=[geo1, geo2],
                                        labels=['geo1', 'geo2'])

        Alternatively, Geometry objects can be after instantiating MutliGeometry

        Example
        -------

            .. code-block:: python

                import dolfin
                geo = MutliGeometry()
                mesh1 = dolfin.UnitSquareMesh(3,3)
                mesh2 = dolfin.UnitSquareMesh(5,5)
                geo1 = Geometry2D(mesh1)
                geo2 = Geometry2D(mesh2)
                geo.add_geometry(geo1)
                geo.add_geometry(geo2)

        Parameters
        ----------
        geometries : list or tuple
            The geometries that should be part of the MultiGeometry.
        lables : list or tuple
            Contains a label for each of the Geometry objects in `geometries`
        """

        self.geometries = {}

        if geometries is None:
            self._geo_type = None
        else:
            self._geo_type = type(geometries[0])

        if geometries is not None:
            if len(geometries) != len(labels):
                msg = "Lists geometries and labels must have the same length."
                raise ValueError(msg)

            for geometry, l in zip(geometries, labels):
                if not isinstance(geometry, self._geo_type):
                    msg = "Can only add geometries of the same type. You tried to add instances of {} and {}".format(type(geo), self._geo_type)
                    raise TypeError(msg)
                self.geometries[l] = geometry


    @classmethod
    def _load_from_file(cls, h5name, h5group, comm, geometry_type=Geometry):
        if h5name is not list:
            return geometry_type._load_from_file(h5name, h5group, comm)
        for h in h5name:
            return geometry_type._load_from_file(h, h5group, comm)


    def copy(self, deepcopy):
        if not self.geometries:
            msg = "The MultiGeometry is empty."
            raise KeyError(msg)

        new_geometries = {}
        for key, value in self.geometries:
            new_geometries[key] = value._copy(deepcopy)

        return new_geometries


    def add_geometry(self, geometry, label):
        """Adds a Geometry object to the MultiGeometry. Geometry object must be
        of the same type as other Geometry objects contained in the MultiGeometry.

        Parameters
        ----------
        geometry : :class:`geometry.Geometry`
            The geometry to add to the MultiGeometry.
        label : str
            A label for the geometry.
        """

        # Check that correct instance of Geometry is added if self.geometries
        # already contains geometries
        if self._geo_type is not None and not isinstance(geometry, self._geo_type):
            msg = "Can only add instances of Geometry. You tried to add an instance of {}".format(type(geometry))
            raise TypeError(msg)

        if label in self.geometries.values():
            msg = "The label '{}' already exists."
            raise KeyError(msg)

        # If this is the first geometry set self._geo_type
        if self._geo_type is None:
            self._geo_type = type(geometry)

        self.geometries[label] = geometry
