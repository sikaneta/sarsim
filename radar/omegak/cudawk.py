# -*- coding: utf-8 -*-
import numpy as np
from numba import cuda
from common.utils import FFT_freq
from scipy.constants import c
import math
# import omegak.nbomegak as nbwk
from omegak.stolz_cuda import getInterpolationPoints as gIP
from common.cxinterp import fdiri, inner_fsum

#%% SAR process the signal. Many assumptions are made about the signal
def wkProcessNumbaCuda(procData, 
                       r_sys,
                       ks = None,
                       r = None, 
                       os_factor = 8, 
                       mem_cols = 1024, 
                       mem_rows = 8192, 
                       tempFile = None,
                       Mc = 32):
    """ 
    mem_cols are the number of columns to process at one go. The data are
    organised by
            az1 az2 az3 ...
    rng1
    rng2
    rng2
    .
    .
    .
    
    Thus, mem_cols defines the number of azimuth samples to process in one 
    call
    """
    
    try:
        r = r or np.linalg.norm(r_sys.C.R)
    except AttributeError:
        print("will use center range for w-k processing")
        r = r_sys.mid_range
        
    """ Define the ks frequencies if not already defined """
    if ks is None:
        ks = r_sys.ks_full
        
    """ Calculate the inverse frequency offset for oversampling in
        the frequency signal (this is done in the interpolation step)"""
    # swath = r_sys.far_range - r_sys.near_range
    # baseband_mix = 2.0*np.pi*np.mod(r_sys.mid_range, swath)/swath
    # baseband_mix = 2.0*np.pi*0.5/r_sys.Nr
        
    # r2t = 2.0/c
    cols = len(r_sys.kr_sorted)
    rows = len(ks)
    
    """ Allocate memory and calculate indeces for the pulse workspace """
    # Yos_idx = np.round(FFT_freq(cols,cols,0)).astype('int')
    
    dkr = r_sys.kr[1]-r_sys.kr[0]
    
    row_ticks = list(range(0,rows,mem_rows)) + [rows]
    row_spans = [(row_ticks[k], row_ticks[k+1]) 
                 for k in range(len(row_ticks)-1)]
    
    wk_processed = np.zeros((rows,mem_cols), dtype=complex)
    yupkr = 4.0*np.pi/c*(FFT_freq(cols*os_factor, r_sys.fs, 0) + r_sys.f0)
    yupidx = np.argsort(yupkr)
    
    H = fdiri(cols, os_factor)
    
    for span in row_spans:
        print("Processing rows (azimuth) %d to %d..." % span)
        n_rows = span[1]-span[0]
        YY = np.zeros((n_rows, cols), dtype=complex)
    
        iP = np.zeros((n_rows, r_sys.kr_sorted.shape[0]), 
                      dtype=float)
        
        """ Compute the points at which to interpolate """
        print("Computing interpolation points")
        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(iP.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(iP.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        KS = ks[span[0]:span[1]]
        KR = r_sys.kr_sorted
        a2 = float(r_sys.C.a2)
        a3 = float(r_sys.C.a3)
        a4 = float(r_sys.C.a4)
        max_iter = 10
        error_tol = 1e-10
        KS_d = cuda.to_device(KS)
        KR_d = cuda.to_device(KR)
        iP_d = cuda.to_device(iP)

        gIP[blockspergrid, threadsperblock](KS_d, 
                                            KR_d,
                                            iP_d,
                                            float(r),
                                            a2,
                                            a3,
                                            a4,
                                            max_iter,
                                            error_tol)
        
        iP = iP_d.copy_to_host()
        
        """ Computing DFT in range direction """
        XX = (iP - r_sys.kr_sorted[0])/dkr
        Y = np.fft.fft(procData[span[0]:span[1], :], axis=1)
        
        """ Compute which indeces of the filter to use """
        Hidx = FFT_freq(Mc, Mc, 0).astype(int)
        
        print("Interpolating the signal...")
        Y_d = cuda.to_device(Y)
        XX_d = cuda.to_device(XX)
        YY_d = cuda.to_device(YY)
        H_d = cuda.to_device(H)
        Hidx_d = cuda.to_device(Hidx)

        
        inner_fsum[blockspergrid, threadsperblock](Y_d, 
                                                   YY_d,
                                                   XX_d,
                                                   H_d,
                                                   Hidx_d)


        YY = YY_d.copy_to_host()
        # nbwk.interpolatePulsesCxSimple(procData[span[0]:span[1], :], 
        #                                YY,
        #                                XX, 
        #                                os_factor, 
        #                                yupidx)
        # YY = YY[:,r_sys.kridx_inv]
        
        
        """ Re-order the data"""
        iP = iP[:,r_sys.kridx_inv]
    
        #% Phase multiply the signal
        print("Correcting residual phase and applying IFFT...")
        KR = np.matlib.repmat(r_sys.kr, n_rows, 1)
        wkSignal = np.fft.ifft(YY*np.exp(-1j*(iP-KR)*r_sys.r[0]), axis=1)
        wk_processed[span[0]:span[1],:] = wkSignal[:,0:mem_cols]
        print("[DONE]")
        
    return wk_processed