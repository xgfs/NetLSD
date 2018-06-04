# -*- coding: utf-8 -*-

import os
import sys


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='NetLSD',
    version='1.0.0',
    author='Anton Tsitsulin',
    author_email='anton.tsitsulin@hpi.de',
    description='NetLSD descriptors for graphs. Compare and analyze graph structure on multiple levels!',
    packages=['netlsd',],
    url='http://github.com/xgfs/netlsd',
    download_url = 'https://github.com/xgfs/netlsd/archive/0.1.tar.gz',
    license='MIT',
    install_requires=[
        'numpy',
        'scipy'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    long_description=open('README.rst').read(),
)