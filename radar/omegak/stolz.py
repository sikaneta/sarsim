#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 08:11:49 2020

@author: ishuwa
"""
import numpy as np
from numba import jit, njit, prange, complex128, float64, cuda, void

#%% Define the g function
@njit(float64[:](float64[:],
                 float64,
                 float64,
                 float64))
def g(s, a2, a3, a4):
    return np.sqrt(a2 + a3*s + a4*s**2)

#%% Define the inverse function
@njit(float64[:](float64[:],
                 float64,
                 float64,
                 float64))
def gf(y, a2, a3, a4):
    n_iterations = 3
    g0 = g(np.zeros_like(y), a2, a3, a4)
    for k in range(n_iterations):
        g0 = g(y/g0, a2, a3, a4)
        
    return g0


#%% Define the phi function
@njit(float64[:](float64[:],
                 float64,
                 float64,
                 float64,
                 float64,
                 float64,
                 float64,
                 float64))
def kernel(x, k, l, m, r, a2, a3, a4):
    return x**k/(r**2+x**2)**l/gf(x, a2, a3, a4)**m

#%% Define the derivative of the phi function
@njit(float64[:](float64[:],
                 float64,
                 float64,
                 float64,
                 float64,
                 float64,
                 float64,
                 float64))
def dkernel_factored(x, k, l, m, r, a2, a3, a4):
    gv = gf(x,a2,a3,a4)
    return kernel(x,k-1,l,m, r, a2, a3, a4)*(k - 2*l*x**2/(r**2+x**2) 
            - m*(a3*x*gv + 2*a4*x**2)/(2*gv**4+a3*x*gv+2*a4*x**2))
    

#%% Define the function (and derivative of function) to be inverted
@njit(float64[:](float64[:],
                 float64,
                 float64[:],
                 float64[:],
                 float64[:],
                 float64[:],
                 float64[:],
                 float64,
                 float64,
                 float64,
                 float64))
def f(x, ks, p0, p1, p2, p3, p4, r, a2, a3, a4):
    return (p0*kernel(x,1,1,-1,r,a2,a3,a4) + 
            p1*kernel(x,2,1,0,r,a2,a3,a4) + 
            p2*kernel(x,2,1,2,r,a2,a3,a4) +
            p3*kernel(x,3,1,3,r,a2,a3,a4) +
            p4*kernel(x,4,1,4,r,a2,a3,a4) + ks)

#%% The derivative functin for the Newton iteration
@njit(float64[:](float64[:],
                 float64[:],
                 float64[:],
                 float64[:],
                 float64[:],
                 float64[:],
                 float64,
                 float64,
                 float64,
                 float64))
def df(x, p0, p1, p2, p3, p4, r, a2, a3, a4):
    return (p0*dkernel_factored(x,1,1,-1,r,a2,a3,a4) + 
            p1*dkernel_factored(x,2,1,0,r,a2,a3,a4) + 
            p2*dkernel_factored(x,2,1,2,r,a2,a3,a4) +
            p3*dkernel_factored(x,3,1,3,r,a2,a3,a4) +
            p4*dkernel_factored(x,4,1,4,r,a2,a3,a4))


#%% Compute the interpolation points
@njit(void(float64[:],
           float64[:],
           float64[:,:],
           float64,
           float64,
           float64,
           float64,
           float64,
           float64))
def getInterpolationPoints(KS, 
                           KR,
                           iP,
                           r, 
                           a2,
                           a3,
                           a4, 
                           max_iter = 5, 
                           error_tol = 1e-6):
    
    num_azi_samples = len(KS)#.shape[0]
    for l in prange(num_azi_samples):
        ks = KS[l]
        #% Get the initial guess
        X_n = r*ks/np.sqrt(a2*KR**2-ks**2)
        
        # Define some ceofficients
        p0 = r*KR
        p1 = -np.ones(KR.shape)*ks
        p2 = r*(a3/2)*KR
        p3 = r*a4*KR - a3/2*ks
        p4 = -np.ones_like(KR)*a4*ks
        
        
        #% Write to the array
        if ks != 0:
            #% Newton method
            for k in range(max_iter):
                """Calculate the function at current X_n"""
                f_xn = f(X_n, ks, p0, p1, p2, p3, p4, r, a2, a3, a4)
                
                """Calculate the derivative of the function at current X_n"""
                df_xn = df(X_n, p0, p1, p2, p3, p4, r, a2, a3, a4)
                
                X_n = X_n - f_xn/df_xn
                
                max_error = np.max(np.abs(f_xn/df_xn))
                if max_error < error_tol:
                    break
        
        
            iP[l,:] = (r*KR*kernel(X_n, 0, 0.5, 0, r, a2, a3, a4) - 
                       ks*kernel(X_n, 1, 0.5, 1, r, a2, a3, a4))
        else:
            iP[l,:] = r*KR*kernel(X_n, 0, 0.5, 0, r, a2, a3, a4)
            
        
