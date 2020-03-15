#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 09:58:28 2018

@author: sikaneta
"""
import numpy as np
from common.utils import FFT_freq
import time
#from joblib import Parallel, delayed


defaultConfig = u'/home/sikaneta/local/Matlab/Delph2014/sureConfig.xml'

class physical:
    c = 299792458.0
    mass_EARTH = 5.97219e24
    G = 6.67384e-11

#def Kangles(KR, KX, p):
#    snsq = 1.0/KR**2/(1.0+p**2)*((KR**2+KX**2)/2.0 - np.sqrt(((KR**2-KX**2)/2.0)**2 - (p*KR*KX)**2))
#    sn = np.sqrt(snsq)*np.sign(KX)
#    cs = np.sqrt(1.0-snsq)
#    return sn, cs
#
#def phaseR(KR, KX, p):
#    (sn, cs) = Kangles(KR, KX, p)
#    return KR*(cs + p**2/3.0*sn**4/cs**3)
#
#def dphaseRdkr(KR, KX, p):
#    (sn, cs) = Kangles(KR, KX, p)
#    tn = sn/cs
#    dthetadkr = (1.0-p**2*tn**2)/(p**2*tn*(1.0+1.0/cs**2)-cs/sn)
#    return cs + p**2*sn**4/cs**3 - dthetadkr*(sn-p**2*sn**3/3.0/cs**4*(cs**2+3))
#
#def getInterpolationPointsP(KS, kx, p):
#    """ Get the inital geusses """
#    KR = np.sqrt(KS**2+kx**2)
#    
#    """ Run the Newton method """
#    for k in range(10):
#        error = (KS - phaseR(KR,kx,p))/dphaseRdkr(KR,kx,p)
#        if np.max(np.abs(error))<1.0e-6:
#            break
#        else:
#            KR = KR + error
#            
#    return KR, np.max(np.abs(error))

def getInterpolationPointsFull(r_sys, col_span = None, r=None, max_iter = 5, error_tol = 1e-6):
    """ Check for the number of azimuth bins to process """
    if col_span is None:
        col_span = (0, len(r_sys.ks_full))
        
    #% local stuff
    a2 = r_sys.C.a2
    a3 = r_sys.C.a3
    a4 = r_sys.C.a4
    
    #% Define arrays
    [KX, KR] = np.meshgrid(r_sys.ks_full[col_span[0]:col_span[1]], r_sys.kr_sorted)
    
    #% Get the initial guess
    r = r or np.linalg.norm(r_sys.C.R)
#    KR_n = np.sqrt(KR**2+KX**2/a2)
#    print("Using first order approximation")
#    return KR_n
    X_n = r/np.sqrt(a2)*KX/np.sqrt(a2*KR**2-KX**2)
    
    c2 = a2/r**2
    c1 = np.sqrt(c2)
    
    #% Define some constants
    #bt_sqr = r**2/a2
    gamma = -np.sqrt(a2)*a3/2.0
    xi = a2*a4
    
    #% Define the g function
    def g(s):
        return np.sqrt(a2 + a3*s + a4*s**2)
    
    
    #% Define the kernel and derivatives
    def kernel(x,k,l,m):
        return x**k/(r**2+a2*x**2)**l/g(-x)**m
    
    def dkernel_factored(x,k,l,m):
        return kernel(x,k-1,l,m)*(k - 2*a2*l*x**2/(r**2+a2*x**2) + m*(a3/2.0-a4*x)/g(-x)**2)
        
    
    #% Define the function (and derivative of function) to be inverted
    def f(x):
        return np.sqrt(a2)*(kernel(x,1,0.5,-1) + gamma*kernel(x,2,0.5,2) + xi*kernel(x,3,0.5,3))
    
    def df(x):
        return np.sqrt(a2)*(dkernel_factored(x,1,0.5,-1) + gamma*dkernel_factored(x,2,0.5,2) + xi*dkernel_factored(x,3,0.5,3))
    
    def q(x, C):
        return f(x)*(1.0+C*c1*kernel(x,1,0,1)) - C*np.sqrt(1.0+c2*x**2)
    
    def dq(x, C):
        return df(x)*(1.0+C*c1*kernel(x,1,0,1)) + f(x)*C*c1*dkernel_factored(x,1,0,1) - C*a2/r*kernel(x,1,0.5,0)
    
    #% Newton method
    for k in range(max_iter):
        t = time.time()
        print("Calculating the function at current X_n")
        q_xn = q(X_n, KX/KR)
        
        print("Calculating the derivative of the function at current X_n")
        dq_xn = dq(X_n, KX/KR)
        
        X_n = X_n - q_xn/dq_xn
        
        max_error = np.max(np.abs(q_xn/dq_xn))
        
        print("Max error: %0.9f" % max_error)
        
        print("Iteration time")
        print(time.time()-t)
        if max_error < error_tol:
            break
    
    #%
    return (KR+KX*c1/g(-X_n)*X_n)/np.sqrt(1.0+c2*X_n**2)
    #return KR*np.sqrt(1.0+a2/r**2*X_n**2) - KX*np.sqrt(a2)/r*X_n/g(-X_n)

#%% New implementation with correct root finding
def getInterpolationPointsNew(r_sys, 
                              col_span = None, 
                              r=None, 
                              max_iter = 5, 
                              error_tol = 1e-6):
    """ Check for the number of azimuth bins to process """
    if col_span is None:
        col_span = (0, len(r_sys.ks_full))
        
    #% local stuff
    a2 = r_sys.C.a2
    a3 = r_sys.C.a3
    a4 = r_sys.C.a4
    
    #% Define arrays
    [KX, KR] = np.meshgrid(r_sys.ks_full[col_span[0]:col_span[1]], 
                           r_sys.kr_sorted)
    
    #% Get the initial guess
    r = r or np.linalg.norm(r_sys.C.R)
    X_n = r/np.sqrt(a2)*KX/np.sqrt(a2*KR**2-KX**2)
    
    # Define some ceofficients
    p1 = np.sqrt(a2)*r*KR
    p2 = a2*KX
    p3 = -r*a2*a3/2*KR
    p4 = r*a4*a2**(3/2)*KR- a3/2*a2**(3/2)*KX
    p5 = a2**2*a4*KX
    
    #% Define the g function
    def g(s):
        return np.sqrt(a2 + a3*s + a4*s**2)
    
    
    #% Define the kernel and derivatives
    def kernel(x,k,l,m):
        return x**k/(r**2+a2*x**2)**l/g(-x)**m
    
    def dkernel_factored(x,k,l,m):
        return (kernel(x,k-1,l,m)*(k - 2*a2*l*x**2/(r**2+a2*x**2) 
                                   + m*(a3/2.0-a4*x)/g(-x)**2))
        
    
    #% Define the function (and derivative of function) to be inverted
    def f(x):
        return (p1*kernel(x,1,1,-1) + 
                p2*kernel(x,2,1,0) + 
                p3*kernel(x,2,1,2) +
                p4*kernel(x,3,1,3) +
                p5*kernel(x,4,1,4) - KX)
    
    def df(x):
        return (p1*dkernel_factored(x,1,1,-1) + 
                p2*dkernel_factored(x,2,1,0) + 
                p3*dkernel_factored(x,2,1,2) +
                p4*dkernel_factored(x,3,1,3) +
                p5*dkernel_factored(x,4,1,4))
    
    #% Newton method
    for k in range(max_iter):
        t = time.time()
        print("Calculating the function at current X_n")
        f_xn = f(X_n)
        
        print("Calculating the derivative of the function at current X_n")
        df_xn = df(X_n)
        
        X_n = X_n - f_xn/df_xn
        
        max_error = np.max(np.abs(f_xn/df_xn))
        
        print("Max error: %0.9f" % max_error)
        
        print("Iteration time")
        print(time.time()-t)
        if max_error < error_tol:
            break
    
    # Return the intepolation points
    return r*KR*kernel(X_n, 0, 0.5, 0) + KX*np.sqrt(a2)*kernel(X_n, 1, 0.5, 1)
    
#%%    
def getInterpolationPoints(r_sys, col_span=None):
    #% local stuff
    if col_span is None:
        col_span = (0, len(r_sys.ks_full))
    a2 = r_sys.C.a2
    [KX, KR] = np.meshgrid(r_sys.ks_full[col_span[0]:col_span[1]], r_sys.kr_sorted)
    
    """ Get the inital geusses """
#    KR = np.sqrt(KS**2+kx**2/a2)
    
    return np.sqrt(KR**2+KX**2/a2)
#    """ Run the Newton method """
#    for k in range(10):
#        error = (KS - phaseR(KR,kx,p))/dphaseRdkr(KR,kx,p)
#        if np.max(np.abs(error))<1.0e-6:
#            break
#        else:
#            KR = KR + error
#            
#    return KR, np.max(np.abs(error))

#%% Interpolate the rows of a matrix using FFT oversampling and linear interpolation
def interpolateCx(X, xold, xnew, oversample, osxold):
    # This function will oversample the rows of the matrix X and return the (then) linearly interpolated values in the matrix
    # xnew which interpolates in the column direction only! The entry in row m, column n in the matrix xnew is a request for
    # interpolated value at row xnew[m,n], column n. Thus, there must be as many columns in xnew as in X. Otherwise an error
    # should be thrown.
    
    (rowsX ,colsX) = X.shape
    (rowsXN, colsXN) = xnew.shape
    invalid = (xnew<np.min(xold)) | (xnew>=np.max(xold))
    xnew[invalid] = 0.0
    if colsX!=colsXN:
        print("Error %d != %d the bumber of columns in X and xnew do not match" % (colsX, colsXN))
        print("Returning nothing")
        return None
    
    # First compute the FFT
    (rows, cols) = X.shape
    
    # Switch to frequency domain and zero pad to oversample by oversample factor
    Y = np.zeros((rows*oversample, cols), dtype=X.dtype)
    idx0 = np.round(FFT_freq(rows,rows,0)).astype('int')
    Y[idx0,:] = np.fft.fft(X, axis=0)
    Y = np.fft.ifft(Y, axis=0)*oversample
    (rowsY, cols) = Y.shape
    
    # Calculate the new interpolation indeces
    xx = xnew*oversample
    
    # calculate the floor, ceiling and fractional indeces
    xx_floor = np.floor(xx).astype('int')
    xx_ceil = np.ceil(xx).astype('int')
    xx_fraction = xx - xx_floor
    
    # Compute the linearly interpolated value=
    col_IDX = np.matlib.repmat(np.arange(cols),rowsX,1)
    X_interpolated = (1.0-xx_fraction)*Y[xx_floor, col_IDX] + xx_fraction*Y[xx_ceil, col_IDX] 
    X_interpolated[invalid] = 0.0
    return X_interpolated

#%% Define a function that canplit a sequence into chunks
def chunkify(a, chunk):
    idxs = list(range(0,len(a), chunk)) + [len(a)]
    s = [idxs[d] for d in range(0,len(idxs)-1)] 
    e = [idxs[d] for d in range(1,len(idxs))] 
    return [(ss,ee) for ss,ee in zip(s,e)]

#%%
def interpolateCxIntMem(X, xnew, col_CHUNK, oversample=8):
    # Interpolate the rows of a matrix using FFT oversampling and linear interpolation
    (rowsX ,colsX) = X.shape
    (rowsXN, colsXN) = xnew.shape
    #invalid = (xnew<0.0) | (xnew>=(rowsX-1))
    #xnew[invalid] = 0.0
    if colsX!=colsXN:
        print("Error %d != %d the number of columns in X and xnew do not match" % (colsX, colsXN))
        print("Returning nothing")
        return None
    
    # Define the output
    Y = np.zeros((rowsXN, colsXN), dtype=np.complex64)
    
    # Go through each chunk and interpolate
    for c in chunkify(range(colsX), col_CHUNK):
        Y[:,c[0]:c[1]] = interpolateCxInt(X[:,c[0]:c[1]], xnew[:,c[0]:c[1]], oversample)
        
    # Return
    return Y
    
#%%    
def interpolateCxInt(X, xnew, oversample=8):
    # This function will oversample the rows of the matrix X and return the (then) linearly interpolated values in the matrix
    # xnew which interpolates in the column direction only! The entry in row m, column n in the matrix xnew is a request for
    # interpolated value at row xnew[m,n], column n. Thus, there must be as many columns in xnew as in X. Otherwise an error
    # should be thrown.
    
    (rows ,cols) = X.shape
    (rowsXN, colsXN) = xnew.shape
    invalid = (xnew<0.0) | (xnew>(rows-1))
    xnew[invalid] = 0.0
    if cols!=colsXN:
        print("Error %d != %d the number of columns in X and xnew do not match" % (cols, colsXN))
        print("Returning nothing")
        return None
    
    # Switch to frequency domain and zero pad to oversample by oversample factor
    Y = np.zeros((rows*oversample, cols), dtype=np.complex64)
    idx0 = np.round(FFT_freq(rows,rows,0)).astype('int') #new
    Y[idx0,:] = np.fft.fft(X, axis=0)
    Y = np.fft.ifft(Y, axis=0)*oversample
    (rowsY, cols) = Y.shape
    
    # Calculate the new interpolation indeces
    xx = xnew*oversample
    
    # calculate the floor, ceiling and fractional indeces
    xx_floor = np.floor(xx).astype('int')
    xx_ceil = np.ceil(xx).astype('int')
    xx_fraction = xx - xx_floor
    
    # Compute the linearly interpolated value=
    col_IDX = np.matlib.repmat(np.arange(cols),rows,1)
    X_interpolated = (1.0-xx_fraction)*Y[xx_floor, col_IDX] + xx_fraction*Y[xx_ceil, col_IDX] 
    X_interpolated[invalid] = 0.0
    return X_interpolated
    
    
#%%
def getInterpolationPointsSeptember(r_sys, r=None, col_span = None, max_iter = 5, error_tol = 1e-6):
    """ Check for the number of azimuth bins to process """
    if col_span is None:
        col_span = (0, len(r_sys.ks_full))
        
    #% local stuff
    a2 = r_sys.C.a2
    a3 = r_sys.C.a3
    a4 = r_sys.C.a4
    
    #% Define arrays
    [KX, KR] = np.meshgrid(r_sys.ks_full[col_span[0]:col_span[1]], r_sys.kr_sorted)
    
    #% Get the initial guess
    r = r or np.linalg.norm(r_sys.C.R)
    KR_n = np.sqrt(KR**2+KX**2/a2)
    print("Using first order approximation")
    return KR_n
    X_n = r/np.sqrt(a2)*KX/np.sqrt(a2*KR**2-KX**2)
    
    #% Define some constants
    #bt_sqr = r**2/a2
    gamma = -np.sqrt(a2)*a3/2.0
    xi = a2*a4
    
    #% Define the g function
    def g(s):
        return np.sqrt(a2 + a3*s + a4*s**2)
    
    
    #% Define the kernel and derivatives
    def kernel(x,k,l,m):
        return x**k/(r**2+a2*x**2)**l/g(-x)**m
    
#    def dkernel(x,k,l,m):
#        return (k*kernel(x,k-1,l,m) 
#                - 2.0*a2*l*kernel(x,k+1,l+1,m) 
#                + a3*m/2.0*kernel(x,k,l,m+2)
#                -a4*m*kernel(x,k+1,l,m+2))
    
    def dkernel_factored(x,k,l,m):
        return kernel(x,k-1,l,m)*(k - 2*a2*l*x**2/(r**2+a2*x**2) + m*(a3/2.0-a4*x)/g(-x)**2)
        
    
    #% Define the function (and derivative of function) to be inverted
    def f(x, C):
        return kernel(x,1,0.5,-1) + gamma*kernel(x,2,0.5,2) + xi*kernel(x,3,0.5,3) - C/np.sqrt(a2)
    
    def df(x):
        return dkernel_factored(x,1,0.5,-1) + gamma*dkernel_factored(x,2,0.5,2) + xi*dkernel_factored(x,3,0.5,3)
    
    #% Newton method
    for k in range(max_iter):
        t = time.time()
        print("Calculating the function at current X_n")
        f_xn = f(X_n, KX/KR)
        
        print("Calculating the derivative of the function at current X_n")
        df_xn = df(X_n)
        
        X_n = X_n - f_xn/df_xn
        
        max_error = np.max(np.abs(f_xn/df_xn))
        
        print("Max error: %0.9f" % max_error)
        
        print("Iteration time")
        print(time.time()-t)
        if max_error < error_tol:
            break
    
    #%
    return KR*np.sqrt(1.0+a2/r**2*X_n**2) - KX*np.sqrt(a2)/r*X_n/g(-X_n)
  
    
    
    
    
    