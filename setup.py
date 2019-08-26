from setuptools import setup

setup(name='RFIsim',
      version='0.1',
      description='Realistic Radio Frequency Interference (RFI) simulations for radio interferometers',
      url='http://github.com/chrisfinlay/RFIsim',
      author='Chris Finlay',
      author_email='cfinlay@ska.ac.za',
      license='MIT',
      packages=['RFIsim'],
      scripts=['bin/RFIsim'],
      install_requires=['numpy',
                        'dask',
                        'zarr',
                        'skyfield',
                        'spacetrack',
                        'pytz',
                        'tzwhere',
                        'pandas',
                        'yaml'
                       ],
      zip_safe=False)
