""" This library is based on the module pulse.geometry by Henrik Finsberg. It
has been extended to account for arbitrary geometries.
"""

import logging


def dict_to_namedtuple(d, NamedTuple):
    pass


def set_namedtuple_default(NamedTuple, default=None):
    NamedTuple.__new__.__defaults__ = (default,) * len(NamedTuple._fields)
