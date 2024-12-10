#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="claimrobustness",
    version="0.11",
    description="Claim Matching Robustness",
    author="Jabez Magomere",
    author_email="jabezmagomere@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
