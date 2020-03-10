#!/usr/bin/env python3.7
import scipy as sp
import scipy.spatial as spt
import scipy.stats as sts
import numpy as np
import numpy.linalg as nl
import numpy.random as nr
import sys

import argparse
parser = argparse.ArgumentParser('Demonstrate Minkowski additivity.')
args = parser.parse_args()
from emd import OBJ, greedy_primal_dual

# Demonstrate Minkowski additivity 
# Additivity only applies to points with nonnegative entries

nn = 12
dims_aa = [nn,]*10
dims_bb = [nn,]*11

aa = [nr.rand(_) for _ in dims_aa]
aa = [_/sum(_) for _ in aa]

bb = [nr.rand(_) for _ in dims_bb]
bb = [_/sum(_) for _ in bb]

cc=[]
for _a in aa:
    for _b in bb:
        cc += [_a + _b ]

emd_aa = greedy_primal_dual(aa)
emd_bb = greedy_primal_dual(bb)
emd_cc = greedy_primal_dual(cc)
print()
print('Does obj(A) + obj(B) == obj(A+B)?', abs(  emd_aa['primal objective'] +  emd_bb['primal objective'] - emd_cc['dual objective'])<1e-10)
print()
print('A')
print('  |A|:', len(aa))
print('  primal obj:                  ', emd_aa['primal objective'])
print('  dual   obj:                  ', emd_aa['dual objective'])
print('B')
print('  |B|:', len(bb))
print('  primal obj:                  ', emd_bb['primal objective'])
print('  dual   obj:                  ', emd_bb['dual objective'])
print('C = A + B')
print('  |C|:', len(cc))
print('  primal obj:                  ', emd_cc['primal objective'])
print('  dual   obj:                  ', emd_cc['dual objective'])
print('  primal obj(A) + primal obj(B)', emd_aa['primal objective'] + emd_bb['primal objective'])
