""" This library is based on the module pulse.geometry by Henrik Finsberg. It
has been extended to account for arbitrary geometries.
"""
import os
import h5py

from dolfin import (HDF5File, Mesh, MeshFunction, LogLevel,
                    VectorElement, Function, FunctionSpace)
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


def namedtuple_as_dict(named_tuple):
    """Returns an ordered dictionary of the namedtuple object.

    :param namedtuple named_tuple: namedtuple object
    :returns An ordered dictionary version of named_tuple
    :rtype dict
    """
    try:
        return named_tuple._asdict()
    except AttributeError:
        return named_tuple


def set_namedtuple_default(NamedTuple, default=None):
    """Set default values of a namedtuple type. None by default.

    :param namedtuple NamedTuple: namedtuple type
    :param object default: default value
    """

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
        msg = ("Error!\nGroup: '{}' does not exist in file:" "\n{}").format(
            mgroup, h5name
        )

        with h5py.File(h5name) as h:
            keys = h.keys()
        msg += "\nPossible values for the h5group are {}".format(keys)
        raise IOError(msg)

    # Dummy class to store attributes in
    class Geometry(object):
        pass

    geo = Geometry()

    with HDF5File(comm, h5name, "r") as h5file:

        # Load mesh
        mesh = Mesh(comm)
        h5file.read(mesh, mgroup, True)
        geo.mesh = mesh

        # Get mesh functions
        meshfunctions = ["vfun", "efun", "ffun", "cfun"]\
            if mesh.topology().dim() == 3 else ["vfun", "ffun", "cfun"]

        for dim, attr in enumerate(meshfunctions):
            dgroup = "{}/mesh/meshfunction_{}".format(ggroup, dim)
            mf = MeshFunction("size_t", mesh, dim, mesh.domains())

            if h5file.has_dataset(dgroup):
                h5file.read(mf, dgroup)
                setattr(geo, attr, mf)


        load_local_basis(h5file, lgroup, mesh, geo)
        load_microstructure(h5file, fgroup, mesh, geo, include_sheets)

        # Load the boundary markers
        markers = load_markers(h5file, mesh, ggroup, dgroup)
        geo.markers = markers

        origmeshgroup = "{}/original_geometry".format(h5group)
        if h5file.has_dataset(origmeshgroup):
            original_mesh = Mesh(comm)
            h5file.read(original_mesh, origmeshgroup, True)
            setattr(geo, "original_geometry", original_mesh)

    for attr in meshfunctions:
        if not hasattr(geo, attr):
            setattr(geo, attr, None)

    for attr in (["f0", "s0", "n0", "r0", "c0", "l0"]):
        if not hasattr(geo, attr):
            setattr(geo, attr, None)

    return geo


def save_geometry_to_h5(mesh, h5name, h5group="", markers=None,
                        markerfunctions={}, microstructure={}, local_basis={},
                        comm=mpi_comm_world, other_functions={},
                        other_attributes={}, overwrite_file=False,
                        overwrite_group=True):
    """
    Save geometry and other geometrical functions to a HDF file.

    Parameters
    ----------
    mesh : :class:`dolfin.mesh`
        The mesh
    h5name : str
        Path to the file
    h5group : str
        Folder within the file. Default is "" which means in
        the top folder.
    markers : dict
        A dictionary with markers. See `get_markers`.
    fields : list
        A list of functions for the microstructure
    local_basis : list
        A list of functions for the crl basis
    meshfunctions : dict
        A dictionary with keys being the dimensions the the values
        beeing the meshfunctions.
    comm : :class:`dolfin.MPI`
        MPI communicator
    other_functions : dict
        Dictionary with other functions you want to save
    other_attributes: dict
        Dictionary with other attributes you want to save
    overwrite_file : bool
        If true, and the file exists, the file will be overwritten (default: False)
    overwrite_group : bool
        If true and h5group exist, the group will be overwritten.

    """
    h5name = os.path.splitext(h5name)[0] + ".h5"

    assert isinstance(mesh, Mesh)

    file_mode = "a" if os.path.isfile(h5name) and not overwrite_file else "w"

    # IF we should append the file but overwrite the group we need to
    # check that the group does not exist. If so we need to open it in
    # h5py and delete it.
    if file_mode == "a" and overwrite_group and h5group != "":
        check_h5group(h5name, h5group, delete=True, comm=comm)

    with HDF5File(comm, h5name, file_mode) as h5file:

        # Save mesh
        ggroup = "{}/geometry".format(h5group)
        mgroup = "{}/mesh".format(ggroup)
        h5file.write(mesh, mgroup)

        # Save markerfunctions
        df.begin(LogLevel.PROGRESS, "Saving marker functions.")
        for dim, key in enumerate(markerfunctions.keys()):
            mf = markerfunctions[key]
            if mf is not None:
                dgroup = "{}/mesh/meshfunction_{}".format(ggroup, dim)
                h5file.write(mf, dgroup)
        df.end()

        # Save markers
        df.begin(LogLevel.PROGRESS, "Saving markers.")
        for name, (marker, dim) in markers.items():
            for key_str in ["domain", "meshfunction"]:
                dgroup = "{}/mesh/{}_{}".format(ggroup, key_str, dim)

                if h5file.has_dataset(dgroup):
                    aname = "marker_name_{}".format(name)
                    h5file.attributes(dgroup)[aname] = marker
        df.end()

        # Save microstructure
        df.begin(LogLevel.PROGRESS, "Saving microstructure.")
        for key in microstructure.keys():
            ms = microstructure[key]
            if ms is not None:
                name = "_".join(filter(None, [str(ms), key]))
                fsubgroup = "{}/{}".format(fgroup, name)
                h5file.write(microstructure[key], fsubgroup)
                h5file.attributes(fsubgroup)["name"] = key
                elm = ms.function_space().ufl_element()

        try:
            family, degree = elm.family(), elm.degree()
            fspace = "{}_{}".format(family, degree)
            h5file.attributes(fgroup)["space"] = fspace
            h5file.attributes(fgroup)["names"] = ":".join(microstructure.keys())
        except:
            pass
        df.end()

        # Save local basis
        df.begin(LogLevel.PROGRESS, "Saving local basis.")
        for key in local_basis.keys():
            ml = local_basis[key]
            if ml is not None:
                lgroup = "{}/local basis functions".format(h5group)
                h5file.write(ml, lgroup + "/{}".format(key))
                elm = ml.function_space().ufl_element()

        try:
            family, degree = elm.family(), elm.degree()
            lspace = "{}_{}".format(family, degree)
            h5file.attributes(lgroup)["space"] = lspace
            h5file.attributes(lgroup)["names"] = ":".join(local_basis.keys())
        except:
            pass
        df.end()

        # Save other functions
        df.begin(LogLevel.PROGRESS, "Saving other functions")
        for key in other_functions.keys():
            mo = other_functions[key]
            if mo is not None:
                fungroup = "/".join([h5group, key])
                h5file.write(mo, fungroup)
                elm = mo.function_space().ufl_element()

        try:
            family, degree, vsize = elm.family(), elm.degree(), elm.value_size()
            fspace = "{}_{}".format(family, degree)
            h5file.attributes(fungroup)["space"] = fspace
            h5file.attributes(fungroup)["value_size"] = vsize
        except:
            pass
        df.end()

        # Save other attributes
        df.begin(LogLevel.PROGRESS, "Saving other attributes")
        for key in other_attributes:
            if isinstance(other_attributes[key], str) and isinstance(key, str):
                h5file.attributes(h5group)[key] = other_attributes[key]
            else:
                begin(df.LogLevel.WARNING,
                    "Invalid attribute {} = {}".format(key,
                    other_attributes[key]))
                end()
        df.end()

    df.begin(df.LogLevel.INFO, "Geometry saved to {}".format(h5name))
    df.end()


def check_h5group(h5name, h5group, delete=False, comm=mpi_comm_world):

    if not isinstance(h5group, str):
        msg = "Error! h5group has to be of type string. Your h5group is of type {}".format(type(h5group))
        raise TypeError(msg)

    h5group_in_h5file = False
    if not os.path.isfile(h5name):
        return False

    filemode = "a" if delete else "r"
    if not os.access(h5name, os.W_OK):
        filemode = "r"
        if delete:
            df.begin(df.LogLevel.WARNING,
                "You do not have write access to file " "{}".format(h5name))
            delete = False
            df.end()

    with open_h5py(h5name, filemode, comm) as h5file:
        if h5group in h5file:
            h5group_in_h5file = True
            if delete:
                if parallel_h5py:
                    df.begin(df.LogLevel.DEBUG,
                            "Deleting existing group: " "'{}'".format(h5group))
                    del h5file[h5group]
                    df.end()

                else:
                    if df.MPI.rank(comm) == 0:
                        df.begin(df.LogLevel.DEBUG,
                            "Deleting existing group: " "'{}'".format(h5group))
                        del h5file[h5group]
                        df.end()

    return h5group_in_h5file


def open_h5py(h5name, file_mode="a", comm=mpi_comm_world):

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

        elm = VectorElement(family=family, cell=mesh.ufl_cell(),
                                    degree=int(order), quad_scheme="default")
        V = FunctionSpace(mesh, elm)

        for name in names:
            lb = Function(V, name=name)

            h5file.read(h5file, lb, lgroup + "/{}".format(name))
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

        elm = VectorElement(family=family, cell=mesh.ufl_cell(),
                                    degree=int(order), quad_scheme="default")
        V = FunctionSpace(mesh, elm)

        attrs = ["f0", "s0", "n0"]
        for i, name in enumerate(names):
            func = Function(V, name=name)
            fsubgroup = fgroup + "/{}".format(name)

            h5file.read(func, fsubgroup)

            setattr(geo, attrs[i], func)


def load_markers(h5file, mesh, ggroup, dgroup):
    markers = {}
    try:
        for dim in range(mesh.topology().dim()+1):
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
        begin(LogLevel.INFO, "Unable to load makers")
        markers = get_markers()
        end()

    return markers
