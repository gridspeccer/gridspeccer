 #!/usr/bin/env python

from setuptools import setup, find_packages

version = "0.1.0pre"

setup(name='Gridspeccer',
      version=version,
      description='Helper scripts to organize multi-figure plots.',
      author='Oliver Breitwieser',
      author_email='oliver.breitwieser@kip.uni-heidelberg.de',
      url='https://github.com/obreitwi/gridspeccer',
      packages=["gridspec"],
      package_dir={
          "gridspec": "src/gridspec",
          },
      package_data={
          "gridspec": ["defaults/matplotlibrc"],
          },
      entry_points={
          "console_scripts": [
              "gridspeccer = gridspeccer.cli:main"
          ]},
      license="GNUv3",
      zip_safe=True,
      install_requires=["matplotlib"],
      )
