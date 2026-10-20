#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 08:11:49 2020

@author: ishuwa
"""
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import math
from time import perf_counter
 
#%%
def fdiri(N, oversample):
    n = np.arange(N)
    return np.array([np.fft.fft(np.exp(-1j*2*np.pi*n/N*k/oversample)) 
                     for k in range(oversample)])/N
   
#%% Function to return the indeces of a DFT
def FFT_freq(N, fp, f0):
    """
    Find the appropriate frequencies for a DFT dataset
    
    Due to the aliasing nature of discrete sampling and the "butterfly"
    effect of the DFT operation, it is sometimes difficult to relate a
    value in the array of a DFT transformed sequence to the related pysical
    frequency value. This function computes these values. This function
    replaces the *limited* and often incorrect use of fftshift.

    Parameters
    ----------
    N : int
        The number of samples in the data sequence.
    fp : float
        The physical sampling frequency.
    f0 : float
        The centre frequncy of the "unaliased" real sequence.

    Returns
    -------
    ndarray
        The real frequency values in each index of the related DFT transformed
        sequence.

    """
    freq = np.arange(N, dtype=int)
    fidx = freq>=(N/2)
    freq[fidx] -= N
    unwrapped_offset = np.round(N*f0/fp).astype(int)
    wrapped_offset = unwrapped_offset%N
    cycle = np.round((unwrapped_offset - wrapped_offset)/N).astype(int)
    
    freq = np.roll(freq, wrapped_offset) + wrapped_offset + cycle*N
    return freq*fp/N

#%% Do the inner sum on the GPU
@cuda.jit
def inner_fsum(Y, YY, XX, H, Hidx):
    """
    Given Y[row,:], this function interpolates values to yeild 
    YY[row,:] = Y[XX[row,:]]. It is assumed that along axis=1, data are in 
    the frequency domain
    
    XX[row,:] are floating point numbers corresponding to sample numbers. For
    instance, if XX[row,:] = range(n_rng), then Y[row,:] = YY[row, :]
    
    The filters in H are used for the interpolation. Indeed, each row of H
    should correspond to a time-domain phase ramp that shifts data in the 
    frequencyvdomain by row/oversample samples. The rows of H are the DFT of 
    the time domain phase ramps. Interpolated values in the frequency domain 
    are computed using a convolution with a truncated version of the filter 
    in H, according to the truncation factor Mc.
    
    For each point in XX[row, col], the appropriate filters in H are selected
    to book-end the sample value, after which a linear interpolation between
    these points is applied to approximate the final value

    Parameters
    ----------
    Y : np.ndarray((n_azi, n_rng), dtype=complex)
        Array of orignal function values.
    YY : np.ndarray((n_azi, n_rng), dtype=complex)
        Array of interpolated values.
    XX : np.ndarray((n_azi, n_rng), dtype=float)
        The col value at which to interpolate Y. It is assumed that Y
        is sampled at X = range(n_rng).
    H : np.ndarray((oversample, n_rng), dtype=complex)
        DESCRIPTION.
    Hidx : np.array(Mc,)
        Indeces of the interpolation filter to convolve.

    Returns
    -------
    None
    """
    
    MY,NY = Y.shape
    MXX, NXX = XX.shape
    MY, NY = Y.shape
    oversample, NH = H.shape

    m,n = cuda.grid(2)

    if m >= MXX or n >= NXX:
        # Quit if (x, y) is outside of valid C boundary
        return

    xx = XX[m,n]
    col = int(xx)
    r_idx = 0
    for r_idx in range(oversample):
        if col+r_idx/oversample > xx:
            break
    if r_idx == oversample - 1:
        l_idx = r_idx
    else:
        l_idx = r_idx - 1

    """ Check if xx is an integer """
    if col == xx:
        YY[m,n] = Y[m,col]
        return
    
    """ Run the convolution """
    l_total = 0.0
    for kk in Hidx:
        l_total += H[l_idx,kk]*Y[m, (col-kk)%NY]
        
    r_total = 0.0
    if l_idx == r_idx:
        r_total = Y[m, ((col+1)%NY)]
    else:
        for kk in Hidx:
            r_total += H[r_idx,kk]*Y[m, (col-kk)%NY]

    weight = (xx - (col + l_idx/oversample))*oversample

    YY[m,n] = (1.0-weight)*l_total + weight*r_total
    
    
#%% Do the interpolation 
def testCUDA_FFT_Interp(Nrng = 10000, 
                        Nazi = 8000, 
                        oversample = 16, 
                        Mc = 128,
                        waveform = {
                            "carrier": 5e9,
                            "bandwidth": 30e6,
                            "duration": 30e-6
                            },
                        fs = 40e6):
    """
    Test Cuda complex interpolation in the frequency domain.
    
    This test function generates a chirp signal of the given duration and 
    computes the FFT of this sampled waveform zero-padded to length Nrng.
    
    The generated freqency domain waveform is replicated Nazi times to
    generate a matrix Nazi x Nrng
    
    Frequencies corresponding to the FFT of the chirp are computed. These are
    squeezed by a factor ranging from 0.7 to 0.9 so that we create a matrix of
    frequencies of size Nazi x Nrng, where the top row contians frequencies
    that are squeezed by 0.7 while the bottom row contains frequencies
    squeezed by 0.9. The squeezing operation is given by the expression
    below, where $s$ corresponds to the squeeze factor, $f(k)$ relates 
    fast-time FFT sample number to a frequency and $f_0$ is the carrier
    frequency
    
    $$
    f'(k) = f_0 + (f(k) - f_0)*s
    $$
    
    The objective is to compute the frequecy response at the squeezed 
    values for every row.
    
    A plot is created for a single randomly-selected row of the computed 
    interpolated frequency response to demonstrate the quality of the
    CUDA interpolation, but the real test is how long the function takes to
    run

    Parameters
    ----------
    Nrng : int, optional
        Number of range samples to generate. The default is 10000.
    Nazi : TYPE, optional
        Number of azimuth samples to generate. The default is 8000.
    oversample : int, optional
        Oversample factor to use in interpolation. The default is 16.
    Mc : int, optional
        Truncated length of the ideal sinc function to use. Smaller values
        increase speed while decreasing accuracy of interpolation
    waveform : dict, optional
        A dictionary with keys carrier, bandwidth and duration to describe
        a chirp signal
    fs : float, optional
        Fast-time sampling rate

    Returns
    -------
    None
    """
    
    waveform['rate'] = waveform['bandwidth']/waveform['duration']
    t = np.arange(Nrng)/fs
    t -= np.mean(t)
    chirp = np.exp(1j*2*np.pi*(waveform['carrier']*t +
                               waveform['rate']/2*t**2))
    chirp[np.abs(t) > waveform['duration']/2] = 0.0
    X = np.tile(np.arange(Nrng), (Nazi,1))
    Y = np.tile(np.fft.fft(chirp), (Nazi, 1))
    
    """ Compute the Fast-time frequencies and some new 
        array of frequencies """
    f = waveform['carrier'] + 2*waveform['rate']*t
    squeeze = np.linspace(0.7, 0.9, Nazi)
    A, B = np.meshgrid(f - waveform['carrier'], squeeze)
    ff = waveform['carrier'] + A*B
    
    """ Compute frequencies in terms of sample numbers """
    XX = (ff - f[0])/(f[1]-f[0])

    """ Compute the interpolating filters """
    H = fdiri(len(t), oversample)
    
    """ Create an array to hold the output """
    YY = np.ones_like(XX, dtype=complex)
    
    """ Compute which indeces of the filter to use """
    Hidx = FFT_freq(Mc, Mc, 0).astype(int)
    
    """ Start a timer """
    t_start = perf_counter()
    
    """ Set up the cuda 2-d array parameters """
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(XX.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(XX.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    """ Copy data to the GPU device """
    Y_d = cuda.to_device(Y)
    XX_d = cuda.to_device(XX)
    YY_d = cuda.to_device(YY)
    H_d = cuda.to_device(H)
    Hidx_d = cuda.to_device(Hidx)
    
    """ Execute the GPU function """
    inner_fsum[blockspergrid, threadsperblock](Y_d, 
                                               YY_d,
                                               XX_d,
                                               H_d,
                                               Hidx_d)
    
    """ Copy processed data back to host """
    YY = YY_d.copy_to_host()
    
    """ End the timer """
    t_end = perf_counter()
    dt = t_end -t_start
    print('Required %0.2f seconds to interpolate %d X %d array' % (dt,
                                                                   Nazi,
                                                                   Nrng))
    
    """ Plot some results """
    row = np.random.randint(0,Nazi-1)
    xx = np.arange(Nrng*oversample)/oversample
    yy = np.fft.fft(np.fft.ifft(Y[row,:]), 
                    oversample*Nrng)
    myfns = {
        "real(z)": np.real,
        "imag(z)": np.imag,
        "|z|": np.abs
    }
    fname = 'real(z)'
    myfn = myfns[fname]
    plt.figure()
    plt.plot(xx, myfn(yy))
    plt.plot(X[row, :], myfn(Y[row, :]),'o')
    plt.plot(XX[row, :], myfn(YY[row, :]),'x')
    plt.legend(['Zero-pad', 'Original', 'Cuda interpolated'])
    plt.title("Interpolated value for pulse %d" % row)
    plt.xlabel('Sample number')
    plt.ylabel(fname)
    plt.grid()
 