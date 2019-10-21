from setuptools import setup
import os
import re

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(name='fenics-geometry',
      version=find_version("geometry", "__init__.py"),
      description='A library handling geometries for Fenics-based problems. Based on pulse.geometry by Henrik Finsberg',
      url='https://github.com/ComputationalPhysiology/fenics-geometry',
      author='Alexandra K. Diem',
      author_email='alexandra@simula.no',
      license='LGPL3',
      packages=['geometry'],
      install_requires=['pytest', 'fenics'],
      zip_safe=False)
