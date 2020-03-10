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
dims = [nn,]*121

aa = [nr.rand(_) for _ in dims]
aa = [_/sum(_) for _ in aa]


rr = nr.rand(nn)*10
bb = [_ + rr for _ in aa]


emd_aa = greedy_primal_dual(aa)
emd_bb = greedy_primal_dual(bb)
print()
print('Does obj(A) == obj(A+rand(n)*10)?', abs(emd_aa['primal objective'] - emd_bb['primal objective'])<1e-10)
print()
print('A')
print('  |A|:', len(aa))
print('  primal obj:                  ', emd_aa['primal objective'])
print('  dual   obj:                  ', emd_aa['dual objective'])
print('B = A + rand(n)*10')
print('  |B|:', len(bb))
print('  primal obj:                  ', emd_bb['primal objective'])
print('  dual   obj:                  ', emd_bb['dual objective'])
