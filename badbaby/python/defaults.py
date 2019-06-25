#!/usr/bin/env python

"""Setup global study parameters."""

__author__ = 'Kambiz Tavabi'
__copyright__ = 'Copyright 2018, Seattle, Washington'
__license__ = 'MIT'
__version__ = '0.1.0'
__maintainer__ = 'Kambiz Tavabi'
__email__ = 'ktavabi@uw.edu'
__status__ = 'Development'

import os.path as op
from pathlib import Path

static = op.join(Path(__file__).parents[1], 'static')
datadir = op.join(Path(__file__).parents[1], 'data')
datapath = '/media/ktavabi/ALAYA/data/ilabs/badbaby'
paradigms = {kk: op.join(datapath, vv)
             for kk, vv in zip(['mmn', 'assr', 'ids'],
                               ['mismatch', 'tone', 'speech'])}
paradigms['exclusion'] = dict()
paradigms['exclusion']['assr'] = ['108',
                                  '925b',
                                  '130a',  # Noisy sensors MF autobad
                                  '304a',  # Noisy sensors MF autobad
                                  '311a',  # Noisy sensors MF autobad
                                  '318a',  # Noisy sensors MF autobad
                                  '117b',  # Noisy sensors MF autobad
                                  '209b',  # Noisy sensors MF autobad
                                  '921a',  # No ECG surrogate
                                  '925a',  # Noisy sensors Autoreject
                                  '208b',  # Noisy sensors Autoreject
                                  '311b',  # No ECG surrogate
                                  '127a',  # Noisy sensors Autoreject
                                  '134a',  # Noisy sensors Autoreject
                                  '229a',  # No events
                                  '316a',
                                  '218a'   # No ECG surrogate
                                  ]
paradigms['exclusion']['mmn'] = []
paradigms['exclusion']['ids'] = []
paradigms['run_nms'] = {kk: vv for kk, vv in zip(['mmn', 'assr', 'ids'],
                                                 ['mmn', 'am', 'ids'])}
# evoked topoplot viz params
ts_args = {'gfp': True}
topomap_args = {'outlines': 'skirt', 'sensors': False}
sensors = {'lh': ['MEG0111', 'MEG0112', 'MEG0113',
                  'MEG0121', 'MEG0122', 'MEG0123',
                  'MEG0341', 'MEG0342', 'MEG0343',
                  'MEG0321', 'MEG0322', 'MEG0323',
                  'MEG0331', 'MEG0332', 'MEG0333',
                  'MEG0131', 'MEG0132', 'MEG0133',
                  'MEG0211', 'MEG0212', 'MEG0213',
                  'MEG0221', 'MEG0222', 'MEG0223',
                  'MEG0411', 'MEG0412', 'MEG0413',
                  'MEG0421', 'MEG0422', 'MEG0423',
                  'MEG0141', 'MEG0142', 'MEG0143',
                  'MEG1511', 'MEG1512', 'MEG1513',
                  'MEG0241', 'MEG0242', 'MEG0243',
                  'MEG0231', 'MEG0232', 'MEG0233',
                  'MEG0441', 'MEG0442', 'MEG0443',
                  'MEG0431', 'MEG0432', 'MEG0433',
                  'MEG1541', 'MEG1542', 'MEG1543',
                  'MEG1521', 'MEG1522', 'MEG1523',
                  'MEG1611', 'MEG1612', 'MEG1613',
                  'MEG1621', 'MEG1622', 'MEG1623',
                  'MEG1811', 'MEG1812', 'MEG1813',
                  'MEG1821', 'MEG1822', 'MEG1823',
                  'MEG1531', 'MEG1532', 'MEG1533',
                  'MEG1721', 'MEG1722', 'MEG1723',
                  'MEG1641', 'MEG1642', 'MEG1643',
                  'MEG1631', 'MEG1632', 'MEG1633',
                  'MEG1841', 'MEG1842', 'MEG1843',
                  'MEG1911', 'MEG1912', 'MEG1913',
                  'MEG1941', 'MEG1942', 'MEG1943',
                  'MEG1711', 'MEG1712', 'MEG1713'],
           'rh': ['MEG1421', 'MEG1422', 'MEG1423',
                  'MEG1411', 'MEG1412', 'MEG1413',
                  'MEG1221', 'MEG1222', 'MEG1223',
                  'MEG1231', 'MEG1232', 'MEG1233',
                  'MEG1241', 'MEG1242', 'MEG1243',
                  'MEG1441', 'MEG1442', 'MEG1443',
                  'MEG1321', 'MEG1322', 'MEG1323',
                  'MEG1311', 'MEG1312', 'MEG1313',
                  'MEG1121', 'MEG1122', 'MEG1123',
                  'MEG1111', 'MEG1112', 'MEG1113',
                  'MEG1431', 'MEG1432', 'MEG1433',
                  'MEG2611', 'MEG2612', 'MEG2613',
                  'MEG1331', 'MEG1332', 'MEG1333',
                  'MEG1341', 'MEG1342', 'MEG1343',
                  'MEG1131', 'MEG1132', 'MEG1133',
                  'MEG1141', 'MEG1142', 'MEG1143',
                  'MEG2621', 'MEG2622', 'MEG2623',
                  'MEG2641', 'MEG2642', 'MEG2643',
                  'MEG2421', 'MEG2422', 'MEG2423',
                  'MEG2411', 'MEG2412', 'MEG2413',
                  'MEG2221', 'MEG2222', 'MEG2223',
                  'MEG2211', 'MEG2212', 'MEG2213',
                  'MEG2631', 'MEG2632', 'MEG2633',
                  'MEG2521', 'MEG2522', 'MEG2523',
                  'MEG2431', 'MEG2432', 'MEG2433',
                  'MEG2441', 'MEG2442', 'MEG2443',
                  'MEG2231', 'MEG2232', 'MEG2233',
                  'MEG2311', 'MEG2312', 'MEG2313',
                  'MEG2321', 'MEG2322', 'MEG2333',
                  'MEG2531', 'MEG2532', 'MEG2533']}
