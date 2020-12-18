#! /usr/bin/env python

import setuptools  # type: ignore[import]

setuptools.setup(
        name='coastal',
        author='Jaime R. Calzada',
        description='DEM to unstructured mesh interpolation in parallel',
        packages=setuptools.find_packages(),
        python_requires='>=3.7',
        install_requires=[
            'tqdm',
            'numpy',
            'scipy',
            'fiona',
            'shapely',
            'pyproj',
            'requests',
            'geopandas',
            'rasterio',
            'matplotlib',
            'wget',
        ],
        entry_points={
            'console_scripts': [
                'coastal=coastal.__main__:main'
                ]
            }

)
