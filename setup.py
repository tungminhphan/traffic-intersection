import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
   name='traffic-intersection',
   version='0.1.2',
   description='This is an example of the design-by-contract approach',
   license="BSD 3-Clause",
   long_description=long_description,
   author='Tung M. Phan',
   author_email='tung@caltech.com',
   url="https://github.com/tungminhphan/traffic-intersection",
   packages=setuptools.find_packages(),
   install_requires=['imageio', 'scipy', 'numpy', 'matplotlib', 'Pillow'], #external packages as dependencies
   include_package_data=True # set this to True in include non .py files like .png
)
