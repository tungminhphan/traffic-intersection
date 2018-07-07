from setuptools import setup

setup(
   name='street_intersection',
   version='0.0.1',
   description='This is an example of the design-by-contract approach',
   license="BSD 3-Clause",
   long_description='This project involves the design and implementation of an autonomous traffic intersection system that can be extended to other road sections.',
   author='Tung M. Phan',
   author_email='tung@caltech.com',
   url="https://github.com/tungminhphan/street_intersection",
   packages=['street_intersection'],  #same as name
   install_requires=['imageio', 'scipy', 'numpy', 'matplotlib', 'Pillow'], #external packages as dependencies
)
