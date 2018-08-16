# -*- coding: utf-8 -*-

"""Docstring"""

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
#
# License: BSD (3-clause)

import os
from expyfun import run_subprocess

cmd = """
from expyfun import assert_version
assert_version('8511a4d')
"""
try:
    run_subprocess(['python', '-c', cmd], cwd=os.getcwd())
except Exception as exp:
    print('Failure: {0}'.format(exp))
else:
    print('Success!')