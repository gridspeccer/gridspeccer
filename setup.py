#!/usr/bin/env python

from setuptools import setup, find_packages

version = "0.1.0pre"

setup(name='Gridspeccer',
      version=version,
      description='Helper scripts to organize multi-figure plots.',
      author='Oliver Breitwieser',
      author_email='oliver.breitwieser@kip.uni-heidelberg.de',
      url='https://github.com/obreitwi/gridspeccer',
      packages=find_packages(include=['gridspeccer', 'gridspeccer.*']),
      entry_points={
          "console_scripts": [
              "gridspeccer = gridspeccer.cli:plot"
          ]},
      license="GNUv3",
      zip_safe=True,
      install_requires=["matplotlib", "scikit-image"],
      )
