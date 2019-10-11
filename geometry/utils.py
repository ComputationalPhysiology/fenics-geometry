""" This library is based on the module pulse.geometry by Henrik Finsberg. It
has been extended to account for arbitrary geometries.
"""

import logging


def dict_to_namedtuple(d, NamedTuple):
    pass


def make_logger(name, level=parameters["log_level"]):
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
