#!/usr/bin/env python3.7
import scipy as sp
import scipy.spatial as spt
import scipy.stats as sts
import numpy as np
import numpy.linalg as nl
import numpy.random as nr
import sys

import argparse
parser = argparse.ArgumentParser('Demonstrate convex monotonicity.')
args = parser.parse_args()
from emd import OBJ, greedy_primal_dual

nn = 5
dims_bb = [nn,]*250

# B is the parent set
bb = [nr.randn(_) + 10 for _ in dims_bb]
bb = [_/sum(_) for _ in bb]

zz = [np.zeros(nn),]
hull_bb = spt.ConvexHull(bb + zz)

aa = [bb[_] for _ in hull_bb.vertices[:-1]]

emd_aa = greedy_primal_dual(aa)
emd_bb = greedy_primal_dual(bb)

# Sanity check
# The polytope zz + bb and zz + aa share extreme vertices
print('COMMENT')
print('  Check for equality, since A and B share extreme vertices')
print()
print('Theory check:')
print('  Is objective A == objective B?', abs(emd_aa['primal objective'] - emd_bb['primal objective'])<1e-10)
print('A')
print('  |A|:', len(aa))
print('  primal obj:                  ', emd_aa['primal objective'])
print('  dual   obj:                  ', emd_aa['dual objective'])
print('B')
print('  |B|:', len(bb))
print('  primal obj:                  ', emd_bb['primal objective'])
print('  dual   obj:                  ', emd_bb['dual objective'])

