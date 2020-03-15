#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:38:29 2020

@author: ishuwa
"""
import numpy as np
import omegak.nbomegak as nbwk

#%% Define some constants
rows = len(r_sys.kr_sorted)
cols = len(r_sys.ks_full)
Yos = np.zeros((rows*os_factor, ), dtype=np.complex128)
Yos_idx = np.round(FFT_freq(rows,rows,0)).astype('int')
YY = np.zeros((rows, cols), dtype=np.complex128)

#%% Calculate the interpolation points
iP = np.zeros((r_sys.kr_sorted.shape[0], span[1]-span[0]), dtype=float)
success = nbwk.getInterpolationPoints(r_sys.ks_full[span[0]:span[1]], 
                                     r_sys.kr_sorted,
                                     iP,
                                     r,
                                     r_sys.C.a2,
                                     r_sys.C.a3,
                                     r_sys.C.a4,
                                     max_iter=5,
                                     error_tol = 1e-5)

#%% Do the interpolation
success = nbwk.interpolatePulseCx(procData[r_sys.kridx,span[0]:span[1]], 
                                  YY[:,span[0]:span[1]],
                                  (iP - r_sys.kr_sorted[0])/dkr, 
                                  Yos, 
                                  Yos_idx)
