#!/usr/bin/env python

"""setup script"""

import os
from setuptools import find_packages, setup

NAME = 'badbaby'
DESCRIPTION = 'speech infants MEG'
URL = 'https://github.com/ktavabi/badbaby'
EMAIL = 'ktavabi@gmail.com'
AUTHOR = 'Kambiz Tavabi'
VERSION = '0.1'
<<<<<<< HEAD
REQUIRED = ['janitor', 'pandas-profiling']
=======
REQUIRED = ['janitor', 'pandas-profiling', 'mne', 'expyfun']
>>>>>>> dd32bca43f8409c17ab1a5c2736460ef6ca05f32

here = os.path.abspath(os.path.dirname(__file__))  # Where the magic happens:
setup(
    name=NAME,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    version=VERSION,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering'
    ]
)
