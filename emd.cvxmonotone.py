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
dims_bb = [nn,]*22

# B is the parent set
bb = [nr.randn(_) for _ in dims_bb]
bb = [_/sum(_) for _ in bb]

# A is convex combination of elements from B
aa = []
while len(aa)<100:
    ll = nr.rand(len(bb))
    ll /=sum(ll)
    aa.append( sum([_l*_b for _l, _b in zip(ll, bb)]))

emd_aa = greedy_primal_dual(aa)
emd_bb = greedy_primal_dual(bb)

# Sanity check
# The polytope zz + bb + aa contains all vertices in zz + bb plus a few more, namely
# all the vertices of aa. Thus, its volume must be at least as large as as the 
# volume of zz + bb. The sanity check tests the opposite inequality.
zz = [range(nn),]
hull_bb = spt.ConvexHull(zz + bb)
hull_aa_union_bb = spt.ConvexHull(zz + bb + aa)
print()
print('SANITY CHECK:')
print('  is conv(A) \subset conv(B)?:', abs(hull_aa_union_bb.volume - hull_bb.volume )<1e-10)
print()
print('COMMENT')
print('  Check for /strict/ inequality, since A is strictly contained in B')
print()
print('Theory check:')
print('  Is objective A < objective B?', emd_aa['primal objective'] < emd_bb['primal objective'])
print('A')
print('  |A|:', len(aa))
print('  primal obj:                  ', emd_aa['primal objective'])
print('  dual   obj:                  ', emd_aa['dual objective'])
print('B')
print('  |B|:', len(bb))
print('  primal obj:                  ', emd_bb['primal objective'])
print('  dual   obj:                  ', emd_bb['dual objective'])

