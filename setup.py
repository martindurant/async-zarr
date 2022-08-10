#!/usr/bin/env python

from setuptools import setup


setup(
    name="azarr",
    version="0.0.1",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    project_urls={},
    maintainer="Martin Durant",
    maintainer_email="mdurant@anaconda.com",
    license="BSD",
    keywords="file",
    packages=["azarr"],
    python_requires=">=3.8",
    install_requires=["zarr"],
    zip_safe=False,
)
