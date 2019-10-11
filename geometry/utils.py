""" This library is based on the module pulse.geometry by Henrik Finsberg. It
has been extended to account for arbitrary geometries.
"""

import logging
import dolfin as df


def dict_to_namedtuple(d, NamedTuple):
    pass


def make_logger(name, level=df.get_log_level()):
    def log_if_process0(record):
        if dolfin.MPI.rank(mpi_comm_world()) == 0:
            return 1
        else:
            return 0

    mpi_filt = Object()
    mpi_filt.filter = log_if_process0

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

    dolfin.set_log_level(logging.WARNING)

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
                            include_sheets=True, comm=df.MPI.comm_world,
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
    if not io_utils.check_h5group(h5name, mgroup, delete=False, comm=comm):
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

    with dolfin.HDF5File(comm, h5name, "r") as h5file:

        # Load mesh
        mesh = dolfin.Mesh(comm)
        io_utils.read_h5file(h5file, mesh, mgroup, True)
        geo.mesh = mesh

        # Get mesh functions
        for dim, attr in zip(range(4), ["vfun", "efun", "ffun", "cfun"]):

            dgroup = "{}/mesh/meshfunction_{}".format(ggroup, dim)
            mf = dolfin.MeshFunction("size_t", mesh, dim, mesh.domains())

            if h5file.has_dataset(dgroup):
                io_utils.read_h5file(h5file, mf, dgroup)
            setattr(geo, attr, mf)

        load_local_basis(h5file, lgroup, mesh, geo)
        load_microstructure(h5file, fgroup, mesh, geo, include_sheets)

        # Load the boundary markers
        markers = load_markers(h5file, mesh, ggroup, dgroup)
        geo.markers = markers

        origmeshgroup = "{}/original_geometry".format(h5group)
        if h5file.has_dataset(origmeshgroup):
            original_mesh = dolfin.Mesh(comm)
            io_utils.read_h5file(h5file, original_mesh, origmeshgroup, True)
            setattr(geo, "original_geometry", original_mesh)

    for attr in ["f0", "s0", "n0", "r0", "c0", "l0", "cfun", "vfun", "efun", "ffun"]:
        if not hasattr(geo, attr):
            setattr(geo, attr, None)

    return geo
