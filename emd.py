#!/usr/bin/env python3.7
import scipy as sp
import scipy.spatial as spt
import scipy.stats as sts
import numpy as np
import numpy.linalg as nl
import numpy.random as nr
import sys

# Implementation of algorithm listed in 
# 
#  @article{KLINE2019128,
#  title = "Properties of the d-dimensional earth mover’s problem",
#  journal = "Discrete Applied Mathematics",
#  volume = "265",
#  pages = "128 - 141",
#  year = "2019",
#  issn = "0166-218X",
#  doi = "https://doi.org/10.1016/j.dam.2019.02.042",
#  url = "http://www.sciencedirect.com/science/article/pii/S0166218X19301441",
#  author = "Jeffery Kline",
#  keywords = "Submodularity, Monge Property, Linear Programming, Greedy Algorithm, Transportation Problem, Convex Polytopes"}

def OBJ(i):
    return max(i) - min(i)

def greedy_primal_dual(aa):
    sum_aa = [sum(_) for _ in aa]
    assert abs(max(sum_aa)-min(sum_aa)) < 1e-10
    AA = [np.copy(_) for _ in aa]

    dims = tuple([len(_) for _ in AA])
    xx = {}
    dual = [np.zeros(d) for d in dims]
    test_dual = [np.zeros(d) for d in dims]
    idx = [0,]*len(AA)
    obj = 0
    while all([i < _ for _, i in zip(dims, idx)]):
        vals = [v[i] for v, i in zip(AA, idx)]
        minval = min(vals)
        i = vals.index(minval)
        xx[tuple(idx)] = minval
        obj += (OBJ(idx)) * minval
        for v, j in zip(AA, idx): v[j] -= minval
        oldidx = np.copy(idx)
        idx[i] += 1
        if idx[i]<dims[i]:
            dual[i][idx[i]] += OBJ(idx) - OBJ(oldidx) + dual[i][idx[i]-1]

    # the above terminates when any entry in idx equals the corresponding value in dims
    # this leaves other dimensions incomplete; the remaining terms of the dual solution 
    # must be filled-in
    for _, i in enumerate(idx):
        try: dual[_][i:] = dual[_][i]
        except: pass

    dualobj =  sum([_.dot(_d) for _, _d in zip(aa, dual)])
    
    return {'x': xx, 'primal objective': obj,
            'dual': dual, 'dual objective': dualobj}

