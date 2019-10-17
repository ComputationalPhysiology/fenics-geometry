from setuptools import setup

setup(name='fenics-geometry',
      version='2019.1.0',
      description='A library handling geometries for Fenics-based problems. Based on pulse.geometry by Henrik Finsberg',
      url='https://github.com/ComputationalPhysiology/fenics-geometry',
      author='Alexandra K. Diem',
      author_email='alexandra@simula.no',
      license='LGPL3',
      packages=['geometry'],
      install_requires=['pytest', 'fenics'],
      zip_safe=False)
