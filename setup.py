# -*- coding: utf-8 -*-

import os
from setuptools import find_packages, setup

# Package meta-static.
NAME = 'badbaby'
DESCRIPTION = 'MEG study of auditory processing in infants.'
URL = 'https://github.com/ktavabi/badbaby'
EMAIL = 'ktavabi@gmail.com'
AUTHOR = 'Kambiz Tavabi'
VERSION = '0.1.0'

# What packages are required for this module to be executed?
REQUIRED = []

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the
# Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))


# Where the magic happens:
setup(
    name=NAME,
    # version=about['__version__'],
    description=DESCRIPTION,
    # long_description=long_description,
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
    ],

)
