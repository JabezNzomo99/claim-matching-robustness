#!/usr/bin/env python

import setuptools
from setuptools import setup

setup(
    name="claimrobustness",
    version="0.1",
    description="Claim Matching Robustness",
    author="Jabez Magomere",
    author_email="jabezmagomere@gmail.com",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
)
