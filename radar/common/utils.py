# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:00:28 2019

@author: SIKANETAI
"""
import numpy as np

#%%
def secondsToDelta(s):
        sign = int(np.sign(s))
        ds = np.abs(s)
        secs = int(ds)
        ds = 1e9*(ds - secs)
        nsecs = int(round(ds))
        secs = sign*secs
        nsecs = sign*nsecs
        return np.timedelta64(secs,'s') + np.timedelta64(nsecs,'ns')

#%%   
def FFT_freq(N, fp, f0):
    freq = np.arange(N, dtype=int)
    fidx = freq>=(N/2)
    freq[fidx] -= N
    unwrapped_offset = np.round(N*f0/fp).astype(int)
    wrapped_offset = unwrapped_offset%N
    cycle = np.round((unwrapped_offset - wrapped_offset)/N).astype(int)
    
    freq = np.roll(freq, wrapped_offset) + wrapped_offset + cycle*N
    return freq*fp/N