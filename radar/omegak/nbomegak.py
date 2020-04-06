#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 08:11:49 2020

@author: ishuwa
"""
import numpy as np
from numba import jit, njit, prange, complex128, float64, cuda

#%% Define the g function
@njit
def g(s, a2, a3, a4):
    return np.sqrt(a2 + a3*s + a4*s**2)

#% Define the kernel and derivatives
@njit
def kernel(x, k, l, m, r, a2, a3, a4):
    return x**k/(r**2+a2*x**2)**l/g(-x, a2, a3, a4)**m

@njit
def dkernel_factored(x, k, l, m, r, a2, a3, a4):
    return (kernel(x,k-1,l,m, r, a2, a3, a4)*(k - 2*a2*l*x**2/(r**2+a2*x**2) 
                               + m*(a3/2.0-a4*x)/g(-x, a2, a3, a4)**2))
    

#% Define the function (and derivative of function) to be inverted
@njit
def f(x, ks, p0, p1, p2, p3, p4, r, a2, a3, a4):
    return (p0*kernel(x,1,1,-1,r,a2,a3,a4) + 
            p1*kernel(x,2,1,0,r,a2,a3,a4) + 
            p2*kernel(x,2,1,2,r,a2,a3,a4) +
            p3*kernel(x,3,1,3,r,a2,a3,a4) +
            p4*kernel(x,4,1,4,r,a2,a3,a4) - ks)

@njit
def df(x, p0, p1, p2, p3, p4, r, a2, a3, a4):
    return (p0*dkernel_factored(x,1,1,-1,r,a2,a3,a4) + 
            p1*dkernel_factored(x,2,1,0,r,a2,a3,a4) + 
            p2*dkernel_factored(x,2,1,2,r,a2,a3,a4) +
            p3*dkernel_factored(x,3,1,3,r,a2,a3,a4) +
            p4*dkernel_factored(x,4,1,4,r,a2,a3,a4))

#%% simplified version
@jit(forceobj=True)
def interpolatePulsesCxSimple(y, YY, Xnew, oversample, yupidx):
    """This function interpolates the values of the matrix Y and writes
    the interpolated values into the matrix YY. Y and YY are matrices
    that represent range in the row direction and pulse in the column
    direction. The matrix Xnew and the sample numbers (double) at which
    to interpolate the original matrix Y. Xnew has the same dimension as
    Y and YY. The vector Yos (os -> oversample) is a workspace in which
    to story temporary oversampled FFT values. while the vector Yos_idx
    provides the indeces for populating this oversampled vector in the
    frequency domain."""
    
    rows, cols  = Xnew.shape
    
    for row in prange(rows):
        Yup = np.fft.fft(y[row,:], cols*oversample)[yupidx]
        
        # Calculate the new interpolation indeces
        invalid = (Xnew[row,:]<0.0) | (Xnew[row,:]>(cols-1))
        #Xnew[row, invalid] = 0.0
        xx = Xnew[row,:]*oversample
        xx[invalid] = 0.0
        
        # calculate the floor, ceiling and fractional indeces
        xx_floor = np.floor(xx).astype('int')
        xx_ceil = np.ceil(xx).astype('int')
        xx_fraction = xx - xx_floor
        
        # Compute the linearly interpolated values
        YY[row,:] = (1.0-xx_fraction)*Yup[xx_floor] + xx_fraction*Yup[xx_ceil] 
        YY[row,invalid] = 0.0
    


@jit(forceobj=True)
def interpolatePulsesCx(Y, YY, Xnew, oversample, Yos_idx, bb_m):
    """This function interpolates the values of the matrix Y and writes
    the interpolated values into the matrix YY. Y and YY are matrices
    that represent range in the row direction and pulse in the column
    direction. The matrix Xnew and the sample numbers (double) at which
    to interpolate the original matrix Y. Xnew has the same dimension as
    Y and YY. The vector Yos (os -> oversample) is a workspace in which
    to story temporary oversampled FFT values. while the vector Yos_idx
    provides the indeces for populating this oversampled vector in the
    frequency domain."""
    
    rows, cols  = Xnew.shape
    bb_factor = np.exp(1j*bb_m*np.arange(cols))
    bb_factor_conj = np.exp(-1j*bb_m*np.arange(cols*oversample)/(oversample))
    
    for row in prange(rows):
        Yup = np.zeros((cols*oversample,), dtype=np.complex128)
        Yup[Yos_idx] = np.fft.fft(bb_factor*Y[row,:])
        Yup = np.fft.ifft(Yup)*oversample*bb_factor_conj
        
        # Calculate the new interpolation indeces
        invalid = (Xnew[row,:]<0.0) | (Xnew[row,:]>(cols-1))
        #Xnew[row, invalid] = 0.0
        xx = Xnew[row,:]*oversample
        xx[invalid] = 0.0
        
        # calculate the floor, ceiling and fractional indeces
        xx_floor = np.floor(xx).astype('int')
        xx_ceil = np.ceil(xx).astype('int')
        xx_fraction = xx - xx_floor
        
        # Compute the linearly interpolated values
        YY[row,:] = (1.0-xx_fraction)*Yup[xx_floor] + xx_fraction*Yup[xx_ceil] 
        YY[row,invalid] = 0.0
    

#%% Compute the interpolation points
@njit#(parallel=True)
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
        X_n = r/np.sqrt(a2)*ks/np.sqrt(a2*KR**2-ks**2)
        
        # Define some ceofficients
        p0 = np.sqrt(a2)*r*KR
        p1 = np.ones(KR.shape)*a2*ks
        p2 = -r*a2*a3/2*KR
        p3 = r*a4*a2**(3/2)*KR- a3/2*a2**(3/2)*ks
        p4 = np.ones(KR.shape)*a2**2*a4*ks
        
        
        
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
        
        #% Write to the array
        iP[l,:] = (r*KR*kernel(X_n, 0, 0.5, 0, r, a2, a3, a4) + 
                   ks*np.sqrt(a2)*kernel(X_n, 1, 0.5, 1, r, a2, a3, a4))
        
    