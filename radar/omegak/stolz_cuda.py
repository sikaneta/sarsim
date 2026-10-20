#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 08:11:49 2020

@author: ishuwa
"""
from numba import cuda
import math

#%% Define the g function
@cuda.jit(device=True)
def g(s, a2, a3, a4):
    return math.sqrt(a2 + a3*s + a4*s**2)

#%% Define the inverse function
@cuda.jit(device=True)
def gf(y, a2, a3, a4):
    n_iterations = 3
    g0 = g(y, a2, a3, a4)
    for k in range(n_iterations):
        g0 = g(y/g0, a2, a3, a4)
        
    return g0


#%% Define the phi function
@cuda.jit(device=True)
def kernel(x, k, l, m, r, a2, a3, a4):
    return x**k/(r**2+x**2)**l/gf(x, a2, a3, a4)**m

#%% Define the derivative of the phi function
@cuda.jit(device=True)
def dkernel_factored(x, k, l, m, r, a2, a3, a4):
    gv = gf(x,a2,a3,a4)
    return kernel(x,k-1,l,m, r, a2, a3, a4)*(k - 2*l*x**2/(r**2+x**2) 
            - m*(a3*x*gv + 2*a4*x**2)/(2*gv**4+a3*x*gv+2*a4*x**2))
    

#%% Define the function (and derivative of function) to be inverted
@cuda.jit(device=True)
def f(x, ks, p0, p1, p2, p3, p4, r, a2, a3, a4):
    return (p0*kernel(x,1,1,-1,r,a2,a3,a4) + 
            p1*kernel(x,2,1,0,r,a2,a3,a4) + 
            p2*kernel(x,2,1,2,r,a2,a3,a4) +
            p3*kernel(x,3,1,3,r,a2,a3,a4) +
            p4*kernel(x,4,1,4,r,a2,a3,a4) + ks)

#%% The derivative functin for the Newton iteration
@cuda.jit(device=True)
def df(x, p0, p1, p2, p3, p4, r, a2, a3, a4):
    return (p0*dkernel_factored(x,1,1,-1,r,a2,a3,a4) + 
            p1*dkernel_factored(x,2,1,0,r,a2,a3,a4) + 
            p2*dkernel_factored(x,2,1,2,r,a2,a3,a4) +
            p3*dkernel_factored(x,3,1,3,r,a2,a3,a4) +
            p4*dkernel_factored(x,4,1,4,r,a2,a3,a4))


#%% Compute the interpolation points
@cuda.jit
def getInterpolationPoints(KS, 
                           KR,
                           iP,
                           r, 
                           a2,
                           a3,
                           a4, 
                           max_iter, 
                           error_tol):
    
    ksidx, kridx = cuda.grid(2)

    if ksidx >= iP.shape[0] or kridx >= iP.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return
    
    # num_azi_samples = len(KS)#.shape[0]
    # for l in prange(num_azi_samples):
    ks = KS[ksidx]
    kr = KR[kridx]
    
    # Define some coefficients
    p0 = r*kr
    p1 = -kr*ks
    p2 = r*(a3/2)*kr
    p3 = r*a4*kr - a3/2*ks
    p4 = -kr*a4*ks
    
    #% Get the initial guess
    X_n = r*ks/math.sqrt(a2*kr**2-ks**2)
    
    #% Write to the array
    if ks != 0:
        #% Newton method
        for k in range(max_iter):
            """Calculate the function at current X_n"""
            f_xn = f(X_n, ks, p0, p1, p2, p3, p4, r, a2, a3, a4)
            
            """Calculate the derivative of the function at current X_n"""
            df_xn = df(X_n, p0, p1, p2, p3, p4, r, a2, a3, a4)
            
            X_n = X_n - f_xn/df_xn
            
            max_error = abs(f_xn/df_xn)
            if max_error < error_tol:
                break
    
    
        iP[ksidx, kridx] = (r*kr*kernel(X_n, 0, 0.5, 0, r, a2, a3, a4) - 
                            ks*kernel(X_n, 1, 0.5, 1, r, a2, a3, a4))
    else:
        iP[ksidx, kridx] = r*kr*kernel(X_n, 0, 0.5, 0, r, a2, a3, a4)
            
        
