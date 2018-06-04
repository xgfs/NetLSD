# -*- coding: utf-8 -*-

import os
import sys


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='NetLSD',
    version='0.0.1',
    description='The funniest joke in the world',
    packages=['netlsd',],
    url='http://github.com/xgfs/netlsd',
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
    long_description=open('README.txt').read(),
)