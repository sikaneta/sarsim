# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:00:28 2019

@author: SIKANETAI
"""
import numpy as np

#%% Function to convert seconds to a timedelta64 object
def secondsToDelta(s):
        sign = int(np.sign(s))
        ds = np.abs(s)
        secs = int(ds)
        ds = 1e9*(ds - secs)
        nsecs = int(round(ds))
        secs = sign*secs
        nsecs = sign*nsecs
        return np.timedelta64(secs,'s') + np.timedelta64(nsecs,'ns')

#%% Function to return the indeces of a DFT
def FFT_freq(N, fp, f0):
    freq = np.arange(N, dtype=int)
    fidx = freq>=(N/2)
    freq[fidx] -= N
    unwrapped_offset = np.round(N*f0/fp).astype(int)
    wrapped_offset = unwrapped_offset%N
    cycle = np.round((unwrapped_offset - wrapped_offset)/N).astype(int)
    
    freq = np.roll(freq, wrapped_offset) + wrapped_offset + cycle*N
    return freq*fp/N

#%% Function to oversample a signal
def upsampleSignal(y, os_factor, k_off=0):
        N = len(y)
        YY = np.zeros((N*os_factor,), dtype=np.complex128)
        Y_idx = np.round(FFT_freq(N, N, k_off)).astype(int)
        YY[Y_idx] = np.fft.fft(y)
        return np.fft.ifft(YY)*os_factor
    

#%% Function to oversample a signal
def upsampleMatrix(y, os_factor, k_off=0):
        N,M = y.shape
        YY = np.zeros((N*os_factor[0],M*os_factor[1]), dtype=y.dtype)
        idx0 = np.round(FFT_freq(N, N, k_off)).astype(int)
        idx1 = np.round(FFT_freq(M, M, k_off)).astype(int)
        YY[tuple(np.meshgrid(idx0, idx1, sparse=True, indexing='ij'))] = np.fft.fft2(y)
        return np.fft.ifft2(YY)*np.prod(os_factor)