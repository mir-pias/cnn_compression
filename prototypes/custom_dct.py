#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy
import matplotlib.pyplot as plt

def dct2(x, norm=None):
    N = len(x)
    n = np.arange(N)

    coeff_vec = []
    
    for k in n:
        yk = 0
        for i in n:
            yk += x[i] * np.cos(np.pi* k*(2*i + 1)/(2*N))
        if norm=='ortho':
            if k == 0:
                yk = np.sqrt(1/(4*N)) * yk
            else:
                yk = np.sqrt(1/(2*N)) * yk
        yk = yk*2
        coeff_vec.append(yk)
        
    return np.array(coeff_vec)






