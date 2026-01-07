# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:40:53 2020

@author: SIKANETAI
"""
import numpy as np
from numba import njit, prange, cuda
import math
from common.utils import FFT_freq, castType

#%%
@njit(parallel=True)
def filterUpsample_2D(array, filter2D, uarray, roundVal):
    """
    Filter and downsample an array
    
    This function has a numba decorator to accelerate computation. This method
    uses parallelization in the loop over columns

    Parameters
    ----------
    array : `numpy.ndarray`
        Input array to be downsampled.
    filter2D : `numpy.ndarray`, (M,N)
        2D interpolation kernel (should be a windowed sinc for Lanczos 
        interpolation)
    uarray : `numpy.ndarray`
        Pre-allocated output array.
    roundVal : `int`
        Int value specifying whether to round numbers. This can be used to
        restrict the output to be consistend with types such as uint8 where 
        values must be between 0 and 255, (this can be done by specifying a 
        roundVal of 255). This is needed because all computations are 
        most likely done in floating point, especially when the supplied 
        filter is in floating point.

    Returns
    -------
    None.

    """
    M,N = filter2D.shape
    dlenM,dlenN = array.shape
    ddlenM,ddlenN = uarray.shape
    usM = int(ddlenM/dlenM)
    usN = int(ddlenN/dlenN)
    mHalf = M//2
    nHalf = N//2
    for m in range(ddlenM):
        for n in prange(ddlenN):
            tmpsum = 0.0 
            data_m_off = np.mod(m-mHalf, ddlenM)
            fm_off = np.mod(usM - np.mod(data_m_off, usM), usM)
            for fm in range(fm_off, M, usM):
                data_m = np.mod(int((data_m_off + fm)/usM), dlenM)
                data_n_off = np.mod(n-nHalf, ddlenN)
                fn_off = np.mod(usN - np.mod(data_n_off, usN), usN)
                for fn in range(fn_off, N, usN):
                    data_n = np.mod(int((data_n_off + fn)/usN), dlenN)
                    tmpsum += filter2D[fm,fn]*array[data_m, data_n]
            if roundVal > 0:
                if (np.real(tmpsum) > roundVal):
                    uarray[m,n] = roundVal
                elif(np.real(tmpsum)<0):
                    uarray[m,n] = 0
                else:
                    uarray[m,n] = tmpsum
            else:
                uarray[m,n] = tmpsum
    return

#%%
@cuda.jit
def filterUpsample_2D_cuda(array, filter2D, uarray, roundVal):
    """
    Filter and downsample an array
    
    This function has a numba decorator to accelerate computation

    Parameters
    ----------
    array : `numpy.ndarray`
        Input array to be downsampled.
    usM : `int`
        Upsample factor in row direction.
    usN : `int`
        Upsample factor in col direction.
    filter2D : `numpy.ndarray`, (M,N)
        2D interpolation kernel
    uarray : `numpy.ndarray`
        Pre-allocated output array.
    roundVal : `int`
        Int value specifying whether to round numbers. This can be used to
        restrict the output to be consistend with types such as uint8 where 
        values must be between 0 and 255, (this can be done by specifying a 
        roundVal of 255). This is needed because all computations are 
        most likely done in floating point, especially when the supplied 
        filter is in floating point.

    Returns
    -------
    None.

    """
    M,N = filter2D.shape
    dlenM,dlenN = array.shape
    ddlenM,ddlenN = uarray.shape
    usM = int(ddlenM/dlenM)
    usN = int(ddlenN/dlenN)
    mHalf = M//2
    nHalf = N//2
    
    m,n = cuda.grid(2)
    
    tmpsum = 0.0 
    data_m_off = (m-mHalf)%(ddlenM)
    fm_off = (usM - data_m_off%usM)%(usM)
    for fm in range(fm_off, M, usM):
        data_m = int((data_m_off + fm)/usM)%(dlenM)
        data_n_off = (n-nHalf)%(ddlenN)
        fn_off = (usN - data_n_off%usN)%(usN)
        for fn in range(fn_off, N, usN):
            data_n = int((data_n_off + fn)/usN)%(dlenN)
            tmpsum += filter2D[fm,fn]*array[data_m, data_n]
    if roundVal > 0:
        if (tmpsum > roundVal):
            uarray[m,n] = roundVal
        elif(tmpsum<0):
            uarray[m,n] = 0
        else:
            uarray[m,n] = tmpsum
    else:
        uarray[m,n] = tmpsum
    uarray[m,n] = tmpsum
        
    return

#%%
def upsample2D(d_array, 
               u_array, 
               order = 3,
               samp_offset = (0.5, 0.5),
               try_cuda = True):
    """
    Upsample a 2D array
    
    This method upsamples a 2D array. It is assumed that the data are to be
    written to u_array (the upsampled array) from values in d_array (the 
    original array). The upsample factors in the row and column directions
    should be the given by the ratio of the size of u_array, to the size of
    d_array, and these should be integers.
    
    The method uses a windowed sinc function
    `Lanczos method <https://en.wikipedia.org/wiki/Lanczos_resampling>` to do 
    the interpolation

    Parameters
    ----------
    d_array : `numpy.ndarray(M,N)`
        The array to be upsampled.
    u_array : `numpy.ndarray(M*usM, N*usN)`
        The upsampled array. This will be written to in the function. An error
        will be thrown if the dimensions of this array are not a multiple of
        the dimensions of d_array
    order : `int`
        The order of the Lanczos interpolation. The default it 3
    samp_offset : `tuple`
        The sample offset for the upsampled data. With the default of
        0.5, 0.5, this means that the sample position for the cell is 
        located in the middle of the cell (in the lower resolution image)
    try_cuda : `bool`
        Flag to inidicate that CUDA GPU processing should be attempted. The
        default is True

    Returns
    -------
    None.

    """
    
    """
    Compute array sizes and upsample factors
    """
    dm,dn = d_array.shape
    um,un = u_array.shape
    usM = int(um/dm)
    usN = int(un/dn)
    
    """
    Compute the interpolation filters
    """
    m = np.arange(0, order, 1/usM) 
    m = np.concatenate([-m[-1:0:-1], m]) + samp_offset[0]
    win_m = np.sinc(m)*np.sinc(m/order)
    n = np.arange(0, order, 1/usN) 
    n = np.concatenate([-n[-1:0:-1], n]) + samp_offset[1]
    win_n = np.sinc(n)*np.sinc(n/order)
    filter2D = np.outer(win_m, win_n)
    
    """
    Choose the processing method and apply
    """
    if cuda.is_available() and try_cuda:
        threadsperblock = (16,16)
        blockspergrid_x = math.ceil(u_array.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(u_array.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        filterUpsample_2D_cuda[blockspergrid, threadsperblock](d_array, 
                                                               filter2D, 
                                                               u_array,
                                                               0)
    else:
        filterUpsample_2D(d_array,
                          filter2D,
                          u_array,
                          0)
        
    return
    

def upsample2DFFT(array, factorM=1, factorN=1, samp_offset = (0.5, 0.5)):
    """
    Upsample an array
    
    Upsample a 2D array. This is useful for upsampling low resolution
    multispectral imagery for pan sharpening. This method uses a zero-padded
    DFT for the upsampling

    Parameters
    ----------
    array : ndarray
        The array to upsample.
    factorM : int, optional
        The amount to increase the sampling frequency in the row direction. 
        The default is 1.
    factorN : int, optional
        The amount by which to increase the sampling frequency in the column
        direction. The default is 1.
    samp_offset : tuple, optional
        A shift (in samples) in the row and columns directions, respectively
        to apply to the upsample imagery so that the data are sampled
        at unaltered (or desired) points in space/time. The default is 
        (0.5, 0.5).

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    (m,n) = array.shape
    uArray = np.zeros((m*factorM, n*factorN), dtype=np.complex128)
    
    idxM = np.round(FFT_freq(m,m,0).astype('int'))
    idxN = np.round(FFT_freq(n,n,0).astype('int'))
    IdxN, IdxM = np.meshgrid(idxN, idxM)
    
    # Zero pad the FFT of the array
    phase = 2.0*np.pi*(samp_offset[0]*IdxM/m + samp_offset[1]*IdxN/n)
    uArray[IdxM, IdxN] = np.fft.fft2(array)*np.exp(-1j*phase)*factorM*factorN
    
    # Return to the time/space domain
    uArray = np.real(np.fft.ifft2(uArray))
        
    
    return castType(uArray, array.dtype)
