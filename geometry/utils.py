""" This library is based on the module pulse.geometry by Henrik Finsberg. It
has been extended to account for arbitrary geometries.
"""

import logging
import os
import h5py

from dolfin import (HDF5File, Mesh, MeshFunction)
import dolfin as df

try:
    import mpi4py
    has_mpi4py = True
except ImportError:
    has_mpi4py = False
    if parallel_h5py:
        raise ImportError
else:
    from mpi4py import MPI as mpi4py_MPI

try:
    import petsc4py
    has_petsc4py = True
except ImportError:
    has_petsc4py = False

mpi_comm_world = df.MPI.comm_world
parallel_h5py = h5py.h5.get_config().mpi


def dict_to_namedtuple(d, NamedTuple):
    pass


def make_logger(name, level=df.get_log_level()):
    def log_if_rank_is_0(record):
        if df.MPI.rank(mpi_comm_world == 0):
            return 1
        else:
            return 0

    # Dummy object
    class Object(object):
        pass

    mpi_filt = Object()
    mpi_filt.filter = log_if_rank_is_0

    logger = logging.getLogger(name)
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(0)
    # formatter = logging.Formatter('%(message)s')
    formatter = logging.Formatter(
        ("%(asctime)s - " "%(name)s - " "%(levelname)s - " "%(message)s")
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addFilter(mpi_filt)

    df.set_log_level(logging.WARNING)

    ffc_logger = logging.getLogger("FFC")
    ffc_logger.setLevel(logging.WARNING)
    ffc_logger.addFilter(mpi_filt)

    ufl_logger = logging.getLogger("UFL")
    ufl_logger.setLevel(logging.WARNING)
    ufl_logger.addFilter(mpi_filt)

    return logger


def set_namedtuple_default(NamedTuple, default=None):
    NamedTuple.__new__.__defaults__ = (default,) * len(NamedTuple._fields)


def load_geometry_from_h5(h5name, h5group="", fendo=None, fepi=None,
                            include_sheets=True, comm=mpi_comm_world,
):
    """Load geometry and other mesh data from
    a h5file to an object.
    If the file contains muliple fiber fields
    you can spefify the angles, and if the file
    contais sheets and cross-sheets this can also
    be included

    :param str h5name: Name of the h5file
    :param str h5group: The group within the file
    :param int fendo: Helix fiber angle (endocardium) (if available)
    :param int fepi: Helix fiber angle (epicardium) (if available)
    :param bool include_sheets: Include sheets and cross-sheets
    :returns: An object with geometry data
    :rtype: object

    """

    logger.info("\nLoad mesh from h5")
    # Set default groups
    ggroup = "{}/geometry".format(h5group)
    mgroup = "{}/mesh".format(ggroup)
    lgroup = "{}/local basis functions".format(h5group)
    fgroup = "{}/microstructure/".format(h5group)

    if not os.path.isfile(h5name):
        raise IOError("File {} does not exist".format(h5name))

    # Check that the given file contains
    # the geometry in the given h5group
    if not check_h5group(h5name, mgroup, delete=False, comm=comm):
        msg = ("Warning!\nGroup: '{}' does not exist in file:" "\n{}").format(
            mgroup, h5name
        )

        with h5py.File(h5name) as h:
            keys = h.keys()
        msg += "\nPossible values for the h5group are {}".format(keys)
        raise IOError(msg)

    # Create a dummy object for easy parsing
    class Geometry(object):
        pass

    geo = Geometry()

    with HDF5File(comm, h5name, "r") as h5file:

        # Load mesh
        mesh = Mesh(comm)
        read_h5file(h5file, mesh, mgroup, True)
        geo.mesh = mesh

        # Get mesh functions
        for dim, attr in zip(range(4), ["vfun", "efun", "ffun", "cfun"]):

            dgroup = "{}/mesh/meshfunction_{}".format(ggroup, dim)
            mf = MeshFunction("size_t", mesh, dim, mesh.domains())

            if h5file.has_dataset(dgroup):
                read_h5file(h5file, mf, dgroup)
            setattr(geo, attr, mf)

        load_local_basis(h5file, lgroup, mesh, geo)
        load_microstructure(h5file, fgroup, mesh, geo, include_sheets)

        # Load the boundary markers
        markers = load_markers(h5file, mesh, ggroup, dgroup)
        geo.markers = markers

        origmeshgroup = "{}/original_geometry".format(h5group)
        if h5file.has_dataset(origmeshgroup):
            original_mesh = Mesh(comm)
            read_h5file(h5file, original_mesh, origmeshgroup, True)
            setattr(geo, "original_geometry", original_mesh)

    for attr in ["f0", "s0", "n0", "r0", "c0", "l0", "cfun", "vfun", "efun", "ffun"]:
        if not hasattr(geo, attr):
            setattr(geo, attr, None)

    return geo


def check_h5group(h5name, h5group, delete=False, comm=df.MPI.comm_world):

    h5group_in_h5file = False
    if not os.path.isfile(h5name):
        return False

    filemode = "a" if delete else "r"
    if not os.access(h5name, os.W_OK):
        filemode = "r"
        if delete:
            logger.warning(
                ("You do not have write access to file " "{}").format(h5name)
            )
            delete = False

    with open_h5py(h5name, filemode, comm) as h5file:
        if h5group in h5file:
            h5group_in_h5file = True
            if delete:
                if parallel_h5py:
                    logger.debug(("Deleting existing group: " "'{}'").format(h5group))
                    del h5file[h5group]

                else:
                    if df.MPI.rank(comm) == 0:
                        logger.debug(
                            ("Deleting existing group: " "'{}'").format(h5group)
                        )
                        del h5file[h5group]

    return h5group_in_h5file


def read_h5file(h5file, obj, group, *args, **kwargs):

    # Hack in order to work with fenics-adjoint
    # if not hasattr(obj, "create_block_variable"):
    #     obj.create_block_variable = lambda: None

    h5file.read(obj, group, *args, **kwargs)


def open_h5py(h5name, file_mode="a", comm=mpi_comm_world()):

    if parallel_h5py:
        if has_mpi4py and has_petsc4py:
            assert isinstance(comm, (petsc4py.PETSc.Comm, mpi4py.MPI.Intracomm))

        if isinstance(comm, petsc4py.PETSc.Comm):
            comm = comm.tompi4py()

        return h5py.File(h5name, file_mode, comm=comm)
    else:
        return h5py.File(h5name, file_mode)


def load_local_basis(h5file, lgroup, mesh, geo):

    if h5file.has_dataset(lgroup):
        # Get local bais functions
        local_basis_attrs = h5file.attributes(lgroup)
        lspace = local_basis_attrs["space"]
        family, order = lspace.split("_")

        namesstr = local_basis_attrs["names"]
        names = namesstr.split(":")

        elm = dolfin.VectorElement(family=family, cell=mesh.ufl_cell(),
                                    degree=int(order), quad_scheme="default")
        V = dolfin.FunctionSpace(mesh, elm)

        for name in names:
            lb = Function(V, name=name)

            read_h5file(h5file, lb, lgroup + "/{}".format(name))
            setattr(geo, name, lb)
    else:
        setattr(geo, "circumferential", None)
        setattr(geo, "radial", None)
        setattr(geo, "longitudinal", None)


def load_microstructure(h5file, fgroup, mesh, geo, include_sheets=True):

    if h5file.has_dataset(fgroup):
        # Get fibers
        fiber_attrs = h5file.attributes(fgroup)
        fspace = fiber_attrs["space"]
        if fspace is None:
            # Assume quadrature 4
            family = "Quadrature"
            order = 4
        else:
            family, order = fspace.split("_")

        namesstr = fiber_attrs["names"]
        if namesstr is None:
            names = ["fiber"]
        else:
            names = namesstr.split(":")

        # Check that these fibers exists
        for name in names:
            fsubgroup = fgroup + "/{}".format(name)
            if not h5file.has_dataset(fsubgroup):
                msg = ("H5File does not have dataset {}").format(fsubgroup)
                logger.warning(msg)

        elm = dolfin.VectorElement(family=family, cell=mesh.ufl_cell(),
                                    degree=int(order), quad_scheme="default")
        V = dolfin.FunctionSpace(mesh, elm)

        attrs = ["f0", "s0", "n0"]
        for i, name in enumerate(names):
            func = Function(V, name=name)
            fsubgroup = fgroup + "/{}".format(name)

            read_h5file(h5file, func, fsubgroup)

            setattr(geo, attrs[i], func)


def load_markers(h5file, mesh, ggroup, dgroup):
    try:
        markers = {}
        for dim in range(mesh.ufl_domain().topological_dimension() + 1):
            for key_str in ["domain", "meshfunction"]:
                dgroup = "{}/mesh/{}_{}".format(ggroup, key_str, dim)

                # If dataset is not present
                if not h5file.has_dataset(dgroup):
                    continue

                def get_attributes():
                    return h5file.attributes(dgroup).list_attributes()

                for aname in get_attributes():
                    if aname.startswith("marker_name"):

                        name = aname.rsplit("marker_name_")[-1]
                        marker = h5file.attributes(dgroup)[
                            "marker_name_{}".format(name)
                        ]
                        markers[name] = (int(marker), dim)

    except Exception as ex:
        logger.info("Unable to load makers")
        logger.info(ex)
        markers = get_markers()

    return markers
