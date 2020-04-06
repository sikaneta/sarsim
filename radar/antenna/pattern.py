#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 09:42:12 2019

@author: ishuwa
"""
from numba import jit, njit, prange, complex128, float64, cuda, int32, void
from scipy.constants import c
import numpy as np
import time
import os

#%%
#def wrapperResponse(rng, u, rad, z):
#    
#    # Find an effective wavelength for the element pattern
#    wavelength = rad['antenna']['wavelength']
#    
#    # Find the minimum antenna length to use in the element factor
#    minAntennaLength = min(rad['antenna']['azimuthLengths'])
#    
#    txDelay = rad["mode"]["txDelay"]
#    rxDelay = rad["mode"]["rxDelay"]
#    txMag = rad["mode"]["txMagnitude"]
#    rxMag = rad["mode"]["rxMagnitude"]
#    aziPositions = rad['antenna']['azimuthPositions']
#    c = cfg.physical.c
#    nearRangeTime = rad['acquisition']['nearRangeTime']
#    return None

#%%   
#@jit(complex128[:](float64[:],
#                   float64,
#                   float64,
#                   float64,
#                   float64[:],
#                   float64[:],
#                   float64[:],
#                   float64[:],
#                   float64[:],
#                   float64,
#                   float64,
#                   float64))
#@jit(nopython=True)
@jit(parallel=True)
def antennaResponse(fastTimes,
                    targetRangeTime, 
                    u, 
                    minAntennaLength,
                    azimuthPositions,
                    txMag,
                    rxMag,
                    txDelay,
                    rxDelay,
                    chirp_bandwidth,
                    chirp_duration,
                    chirp_carrier):
    """ This function takes a range time: targetRangeTime and a look angle: u along
    other parameters which describe the positions of the antenna
    elements and the delays for each element and calculates the response
    to a transmitted signal z.sample(t) """
    
    """ Let's first make sure that there are enough samples in the fastTimes
    array to cover the chirp duration """
    nSamples = len(fastTimes)
    dT = fastTimes[1] - fastTimes[0]
    if (fastTimes[0] <= targetRangeTime) and (fastTimes[-1]>targetRangeTime):
        """ enlarge the fastTimes array as necessary """
        firstValidSample = np.argwhere(fastTimes>=targetRangeTime)[0]
        chirpLengthBuffer = 10
        requiredNumberSamples = np.max([(firstValidSample + 
                                         chirpLengthBuffer + 
                                         np.ceil(chirp_duration/dT)),
                                       len(fastTimes)])
        requiredNumberSamples = 2**int(np.ceil(np.log2(requiredNumberSamples)))
        newFastTimes = np.arange(requiredNumberSamples)*dT + fastTimes[0]
    else:
        return np.zeros(fastTimes.shape, dtype=np.complex128)
    
    # Compute the element factor
    eF = np.sinc(chirp_carrier*minAntennaLength/c*u)**2
    
    # Find where amplitudes are non zero
    idx_tx = txMag > 0.0
    idx_rx = rxMag > 0.0
    
    # Compute transmit times
    tx_delay = (txDelay[idx_tx] + 
                azimuthPositions[idx_tx]*u)
    tx_len = len(tx_delay)
    
    rx_delay = (rxDelay[idx_rx] + 
                azimuthPositions[idx_rx]*u)
    rx_len = len(rx_delay)
    
    # Compute the receive times
    amplitudes = np.outer(txMag[idx_tx], rxMag[idx_rx]).flatten()
    geometric_delay = np.zeros(len(tx_delay)*len(rx_delay), dtype=tx_delay.dtype)
    for k in prange(tx_len):
        for l in range(rx_len):
            geometric_delay[k*tx_len + l] = tx_delay[k] + rx_delay[l]
    
    rate = 1j*np.pi*chirp_bandwidth/chirp_duration
    wcarrier = 2.0*1j*np.pi*chirp_carrier
    
    newnSamples = len(newFastTimes)
    response = np.zeros(newnSamples, dtype=np.complex128)
    for k in prange(newnSamples):
        t = newFastTimes[k] - targetRangeTime - geometric_delay
        idx = (t>=0.0) & (t<chirp_duration)
        if len(idx)>0:
            y = np.exp( wcarrier*t[idx] + rate*(t[idx]-chirp_duration/2.0)**2 )
            response[k] = np.sum(y*amplitudes[idx])
    
    response = response*np.exp( -wcarrier*(newFastTimes) )
    
    # Pulse compress the signal
    baseband_pulse = np.exp( rate*(np.arange(0,chirp_duration,dT) - chirp_duration/2.0)**2 )
    return eF*np.fft.ifft(np.conj(np.fft.fft(baseband_pulse, newnSamples))*np.fft.fft(response))[0:nSamples]

#%% Compute the number of samples that we'll need
def numberWaveformComputationSamples(fastTimes,
                                     targetRangeTime,
                                     waveformDuration,
                                     waveformBufferSamples=10):
    dT = fastTimes[1] - fastTimes[0]
    if (fastTimes[0] <= targetRangeTime) and (fastTimes[-1]>targetRangeTime):
        """ enlarge the fastTimes array as necessary """
        firstValidSample = np.argwhere(fastTimes>=targetRangeTime)[0]
        requiredNumberSamples = np.max([(firstValidSample + 
                                         waveformBufferSamples + 
                                         np.ceil(waveformDuration/dT)),
                                       len(fastTimes)])
        requiredNumberSamples = 2**int(np.ceil(np.log2(requiredNumberSamples)))
        return np.arange(requiredNumberSamples)*dT + fastTimes[0]
    else:
        return None

#%% A number version of the FFT_freq function with scalar frequency
@jit(float64(int32,
             int32, 
             float64,
             float64,),
     nopython=True)
def nbFFT(k,
          N,
          fp,
          f0):
    fk = fp*(f0//fp) + fp*k/N
    return fk if fk < f0+fp/2 else fk - fp

#%% A parallel implementation of the antenna calculation
# @jit(void(complex128[:],    # The frequency delay term (out)
#           complex128[:],    # The dCorr term (out)
#           float64[:],       # Input frequnecy array
#           float64,          # u
#           float64[:],       # azimuth positions
#           float64[:],       # txMag
#           float64[:],       # rxMag
#           float64[:],       # txDelay
#           float64[:]))      # rxDelay
@jit(parallel=True)
def corePatternMultiCPU(freq_delay,
                        dCorrection,
                        freq,
                        u, 
                        azimuthPositions,
                        txMag,
                        rxMag,
                        txDelay,
                        rxDelay):
    """ Calculate where the true time delays and the arrival delay, due to 
        the angle of arrival, should be computed by finding where there is
        an applied gain for the transmit and receive array elements """
    myTxAmp = txMag[txMag > 0.0]
    myTxDly = txDelay[txMag > 0.0]
    myTxPos = azimuthPositions[txMag > 0.0]
    myTxLen = myTxAmp.shape[0]
    
    myRxAmp = rxMag[rxMag > 0.0]
    myRxDly = rxDelay[rxMag > 0.0]
    myRxPos = azimuthPositions[rxMag > 0.0]
    myRxLen = myRxAmp.shape[0]
    
    """ Calculate the pattern delay terms. See written notes """
    pattern_delay = (np.dot(myTxDly, myTxAmp/np.sum(myTxAmp)) 
                    + np.dot(myRxDly, myRxAmp/np.sum(myRxAmp)))
    
    for k in prange(len(freq)):
        w = 1j*2.0*np.pi*freq[k]
        sTx = np.complex(0.0,0.0)
        for kk in range(myTxLen):
            sTx += myTxAmp[kk]*np.exp(-w*(myTxDly[kk] + u*myTxPos[kk]))
            
        sRx = np.complex(0.0,0.0)
        for kk in range(myRxLen):
            sRx += myRxAmp[kk]*np.exp(-w*(myRxDly[kk] + u*myRxPos[kk]))
            
        freq_delay[k] = sTx*sRx
        dCorrection[k] = np.exp(1j*2.0*np.pi*freq[k]*pattern_delay)
    
    return

#%% A wrapper function for the real work done above
def antennaResponseMultiCPU(return_response,
                            fastTimes,
                            targetRangeTime, 
                            u, 
                            minAntennaLength,
                            azimuthPositions,
                            txMag,
                            rxMag,
                            txDelay,
                            rxDelay,
                            chirp_bandwidth,
                            chirp_duration,
                            chirp_carrier):
    """ This function takes a range time: targetRangeTime and a look angle: u along
    other parameters which describe the positions of the antenna
    elements and the delays for each element and calculates the response
    to a transmitted signal z.sample(t) """
    
    """ Let's first make sure that there are enough samples in the fastTimes
    array to cover the chirp duration """
    
    # Compute the element factor
    eF = np.sinc(chirp_carrier*minAntennaLength/c*u)**2
    
    # Ish stuff =======================================
    N = len(fastTimes)#requiredNumberSamples
    f0 = chirp_carrier
    dT = fastTimes[1] - fastTimes[0]
    fp = 1.0/dT
    # freq = np.array([nbFFT(k,N,fp,f0) for k in range(N)])
    
    # Ish stuff =======================================
    freq = np.arange(N, dtype=int)
    fidx = freq>=(N/2)
    freq[fidx] -= N
    unwrapped_offset = np.round(N*f0/fp).astype(int)
    wrapped_offset = unwrapped_offset%N
    cycle = np.round((unwrapped_offset - wrapped_offset)/N).astype(int)
    
    freq = np.roll(freq, wrapped_offset) + wrapped_offset + cycle*N
    freq = freq*fp/N
    
    freq = np.arange(N, dtype=int)
    fidx = freq>=(N/2)
    freq[fidx] -= N
    unwrapped_offset = np.round(N*f0/fp).astype(int)
    wrapped_offset = unwrapped_offset%N
    cycle = np.round((unwrapped_offset - wrapped_offset)/N).astype(int)
    
    freq = np.roll(freq, wrapped_offset) + wrapped_offset + cycle*N
    freq = freq*fp/N
    
    dCorrection = np.zeros(fastTimes.shape, dtype=np.complex128)
    freq_delay = np.zeros_like(dCorrection)
    corePatternMultiCPU(freq_delay,
                        dCorrection,
                        freq,
                        u, 
                        azimuthPositions,
                        txMag,
                        rxMag,
                        txDelay,
                        rxDelay)
    
    rate = 1j*np.pi*chirp_bandwidth/chirp_duration
    wcarrier = 2.0*1j*np.pi*chirp_carrier
    t = fastTimes - targetRangeTime
    
    response = np.fft.ifft(np.fft.fft(np.exp( wcarrier*(t-chirp_duration/2.0) + 
                    rate*(t-chirp_duration/2.0)**2 ))*freq_delay*dCorrection)
    response = response*np.exp( -wcarrier*(fastTimes) )
    
    # Pulse compress the signal
    baseband_pulse = np.exp( rate*(np.arange(0,chirp_duration,dT) 
                                    - chirp_duration/2.0)**2 )
    response = eF*np.fft.ifft(np.conj(np.fft.fft(baseband_pulse, N))
                                      *np.fft.fft(response))
    
    
    for k in range(len(return_response)):
        return_response[k]=response[k]
    
    return

#%% Some old stuff
if False:
    #@jit(parallel=True)
    @jit(void(float64[:],   # fastTimes
              float64,      # targetRangeTime
              float64,      # u
              float64,      # minAntennaLength
              float64[:],   # azimuth positions
              float64[:],   # txMag
              float64[:],   # rxMag
              float64[:],   # txDelay
              float64[:],   # rxDelay
              float64,      # chirp bandwidth
              float64,      # chirp duration
              float64),     # chirp carrier
              parallel=True)
    def antennaResponseMultiCPU2(fastTimes,
                        targetRangeTime, 
                        u, 
                        minAntennaLength,
                        azimuthPositions,
                        txMag,
                        rxMag,
                        txDelay,
                        rxDelay,
                        chirp_bandwidth,
                        chirp_duration,
                        chirp_carrier):
        """ This function takes a range time: targetRangeTime and a look angle: u along
        other parameters which describe the positions of the antenna
        elements and the delays for each element and calculates the response
        to a transmitted signal z.sample(t) """
        
        """ Let's first make sure that there are enough samples in the fastTimes
        array to cover the chirp duration """
        nSamples = len(fastTimes)
        dT = fastTimes[1] - fastTimes[0]
        if (fastTimes[0] <= targetRangeTime) and (fastTimes[-1]>targetRangeTime):
            """ enlarge the fastTimes array as necessary """
            firstValidSample = np.argwhere(fastTimes>=targetRangeTime)[0]
            chirpLengthBuffer = 10
            requiredNumberSamples = np.max([(firstValidSample + 
                                             chirpLengthBuffer + 
                                             np.ceil(chirp_duration/dT)),
                                           len(fastTimes)])
            requiredNumberSamples = 2**int(np.ceil(np.log2(requiredNumberSamples)))
            newFastTimes = np.arange(requiredNumberSamples)*dT + fastTimes[0]
        else:
            return np.zeros(fastTimes.shape, dtype=np.complex128)
        
        
        # Compute the element factor
        eF = np.sinc(chirp_carrier*minAntennaLength/c*u)**2
        
        # Ish stuff =======================================
        N = requiredNumberSamples
        f0 = chirp_carrier
        fp = 1.0/dT
        
        freq = np.arange(N, dtype=int)
        fidx = freq>=(N/2)
        freq[fidx] -= N
        unwrapped_offset = np.round(N*f0/fp).astype(int)
        wrapped_offset = unwrapped_offset%N
        cycle = np.round((unwrapped_offset - wrapped_offset)/N).astype(int)
        
        freq = np.roll(freq, wrapped_offset) + wrapped_offset + cycle*N
        freq = freq*fp/N
        
        """ Calculate where the true time delays and the arrival delay, due to 
            the angle of arrival, should be computed by finding where there is
            an applied gain for the transmit and receive array elements """
        myTxAmp = txMag[txMag > 0.0]
        myTxDly = txDelay[txMag > 0.0]
        myTxPos = azimuthPositions[txMag > 0.0]
        myTxLen = myTxAmp.shape[0]
        
        myRxAmp = rxMag[rxMag > 0.0]
        myRxDly = rxDelay[rxMag > 0.0]
        myRxPos = azimuthPositions[rxMag > 0.0]
        myRxLen = myRxAmp.shape[0]
        
        """ Calculate the pattern delay terms. See written notes """
        pattern_delay = (np.dot(myTxDly, myTxAmp/np.sum(myTxAmp)) 
                        + np.dot(myRxDly, myRxAmp/np.sum(myRxAmp)))
        
        freq_delay = np.zeros(len(freq), dtype=np.complex128)
        num_freq_samples = len(freq)
        for k in prange(num_freq_samples):
            w = 1j*2.0*np.pi*freq[k]
            sTx = np.complex(0.0,0.0)
            for kk in range(myTxLen):
                sTx += myTxAmp[kk]*np.exp(-w*(myTxDly[kk] + u*myTxPos[kk]))
                
            sRx = np.complex(0.0,0.0)
            for kk in range(myRxLen):
                sRx += myRxAmp[kk]*np.exp(-w*(myRxDly[kk] + u*myRxPos[kk]))
                
            freq_delay[k] = sTx*sRx
        
        dCorrection = np.exp(1j*2.0*np.pi*freq*pattern_delay)
        
        rate = 1j*np.pi*chirp_bandwidth/chirp_duration
        wcarrier = 2.0*1j*np.pi*chirp_carrier
        t = newFastTimes - targetRangeTime
        response = np.fft.ifft(np.fft.fft(np.exp( wcarrier*t + 
                                                 rate*(t-chirp_duration/2.0)**2 ))*freq_delay*dCorrection)
        response = response*np.exp( -wcarrier*(newFastTimes) )
        
        # Pulse compress the signal
        baseband_pulse = np.exp( rate*(np.arange(0,chirp_duration,dT) - chirp_duration/2.0)**2 )
        return eF*(np.fft.ifft(np.conj(np.fft.fft(baseband_pulse, 
                                                  len(response)))*np.fft.fft(response))[0:nSamples])


#%% Point the environment variable
if cuda.is_available():
    os.environ['NUMBAPRO_CUDALIB'] = 'C:\\Users\\SIKANETAI\\AppData\\Local\\conda\\conda\\pkgs\\cudatoolkit-9.2-0\\Library\\bin'

    #%%
    @jit(parallel=True)
    def antennaResponseCuda(fastTimes,
                        targetRangeTime, 
                        u, 
                        minAntennaLength,
                        azimuthPositions,
                        txMag,
                        rxMag,
                        txDelay,
                        rxDelay,
                        chirp_bandwidth,
                        chirp_duration,
                        chirp_carrier):
        """ This function takes a range time: targetRangeTime and a look angle: u along
        other parameters which describe the positions of the antenna
        elements and the delays for each element and calculates the response
        to a transmitted signal z.sample(t) """
        
        """ Let's first make sure that there are enough samples in the fastTimes
        array to cover the chirp duration """
        nSamples = len(fastTimes)
        dT = fastTimes[1] - fastTimes[0]
        if (fastTimes[0] <= targetRangeTime) and (fastTimes[-1]>targetRangeTime):
            """ enlarge the fastTimes array as necessary """
            firstValidSample = np.argwhere(fastTimes>=targetRangeTime)[0]
            chirpLengthBuffer = 10
            requiredNumberSamples = np.max([(firstValidSample + 
                                             chirpLengthBuffer + 
                                             np.ceil(chirp_duration/dT)),
                                           len(fastTimes)])
            requiredNumberSamples = 2**int(np.ceil(np.log2(requiredNumberSamples)))
            newFastTimes = np.arange(requiredNumberSamples)*dT + fastTimes[0]
        else:
            return np.zeros(fastTimes.shape, dtype=np.complex128)
        
        
        # Compute the element factor. Normalize by FFT length
        eF = np.sinc(chirp_carrier*minAntennaLength/c*u)**2#/requiredNumberSamples
        
        # Ish stuff =======================================
        N = requiredNumberSamples
        f0 = chirp_carrier
        fp = 1.0/dT
        
        freq = np.arange(N, dtype=int)
        fidx = freq>=(N/2)
        freq[fidx] -= N
        unwrapped_offset = np.round(N*f0/fp).astype(int)
        wrapped_offset = unwrapped_offset%N
        cycle = np.round((unwrapped_offset - wrapped_offset)/N).astype(int)
        
        freq = np.roll(freq, wrapped_offset) + wrapped_offset + cycle*N
        freq = freq*fp/N
        
        """ Calculate where the true time delays and the arrival delay, due to 
            the angle of arrival, should be computed by finding where there is
            an applied gain for the transmit and receive array elements """
        myTxAmp = txMag[txMag > 0.0]
        mydTxDly = txDelay[txMag > 0.0] 
        myuTxDly = u*azimuthPositions[txMag > 0.0]
        myTxDly = mydTxDly + myuTxDly
        myTxLen = myTxAmp.shape[0]
        
        myRxAmp = rxMag[rxMag > 0.0]
        mydRxDly = rxDelay[rxMag > 0.0] 
        myuRxDly = u*azimuthPositions[rxMag > 0.0]
        myRxDly = mydRxDly + myuRxDly
        myRxLen = myRxAmp.shape[0]
        
        """ Calculate the pattern delay terms. See written notes """
        pattern_delay = (np.dot(mydTxDly, myTxAmp/np.sum(myTxAmp)) 
                    + np.dot(mydRxDly, myRxAmp/np.sum(myRxAmp)))
        
        """ Define the output array """
        freq_delay = np.zeros(len(freq), dtype=np.complex128)
        
        threadsperblock = 32
        blockspergrid = (freq_delay.size + (threadsperblock-1))//threadsperblock
        freq_phase[blockspergrid, threadsperblock](freq, myTxAmp, myTxDly, myRxAmp, myRxDly, freq_delay)
        
        dCorrection = np.exp(1j*2.0*np.pi*freq*pattern_delay)
        
        rate = 1j*np.pi*chirp_bandwidth/chirp_duration
        wcarrier = 2.0*1j*np.pi*chirp_carrier
        t = newFastTimes - targetRangeTime
        response = np.fft.ifft(np.fft.fft(np.exp( wcarrier*t + rate*(t-chirp_duration/2.0)**2 ))*freq_delay*dCorrection)
        response = response*np.exp( -wcarrier*(newFastTimes) )
        
        # Pulse compress the signal
        baseband_pulse = np.exp( rate*(np.arange(0,chirp_duration,dT) - chirp_duration/2.0)**2 )
        return eF*np.fft.ifft(np.conj(np.fft.fft(baseband_pulse, len(response)))*np.fft.fft(response))[0:nSamples]
    
    #%%
    @jit(parallel=True)
    def antennaResponseCudaMem(fastTimes,
                        targetRangeTime, 
                        u, 
                        minAntennaLength,
                        azimuthPositions,
                        txMag,
                        rxMag,
                        txDelay,
                        rxDelay,
                        chirp_bandwidth,
                        chirp_duration,
                        chirp_carrier):
        """ This function takes a range time: targetRangeTime and a look angle: u along
        other parameters which describe the positions of the antenna
        elements and the delays for each element and calculates the response
        to a transmitted signal z.sample(t) """
        
        """ Let's first make sure that there are enough samples in the fastTimes
        array to cover the chirp duration """
        nSamples = len(fastTimes)
        dT = fastTimes[1] - fastTimes[0]
        if (fastTimes[0] <= targetRangeTime) and (fastTimes[-1]>targetRangeTime):
            """ enlarge the fastTimes array as necessary """
            firstValidSample = np.argwhere(fastTimes>=targetRangeTime)[0]
            chirpLengthBuffer = 10
            requiredNumberSamples = np.max([(firstValidSample + 
                                             chirpLengthBuffer + 
                                             np.ceil(chirp_duration/dT)),
                                           len(fastTimes)])
            requiredNumberSamples = 2**int(np.ceil(np.log2(requiredNumberSamples)))
            newFastTimes = np.arange(requiredNumberSamples)*dT + fastTimes[0]
        else:
            return np.zeros(fastTimes.shape, dtype=np.complex128)
        
        
        # Compute the element factor. Divide by the FFT length for normalization
        eF = np.sinc(chirp_carrier*minAntennaLength/c*u)**2
        
        # Ish stuff =======================================
        N = requiredNumberSamples
        f0 = chirp_carrier
        fp = 1.0/dT
        
        freq = np.arange(N, dtype=int)
        fidx = freq>=(N/2)
        freq[fidx] -= N
        unwrapped_offset = np.round(N*f0/fp).astype(int)
        wrapped_offset = unwrapped_offset%N
        cycle = np.round((unwrapped_offset - wrapped_offset)/N).astype(int)
        
        freq = np.roll(freq, wrapped_offset) + wrapped_offset + cycle*N
        freq = freq*fp/N
        
        """ Calculate where the true time delays and the arrival delay, due to 
            the angle of arrival, should be computed by finding where there is
            an applied gain for the transmit and receive array elements """
        myTxAmp = txMag[txMag > 0.0]
        mydTxDly = txDelay[txMag > 0.0] 
        myuTxDly = u*azimuthPositions[txMag > 0.0]
        myTxDly = mydTxDly + myuTxDly
        myTxLen = myTxAmp.shape[0]
        
        myRxAmp = rxMag[rxMag > 0.0]
        mydRxDly = rxDelay[rxMag > 0.0] 
        myuRxDly = u*azimuthPositions[rxMag > 0.0]
        myRxDly = mydRxDly + myuRxDly
        myRxLen = myRxAmp.shape[0]
        
        """ Calculate the pattern delay terms. See written notes """
        pattern_delay = (np.dot(mydTxDly, myTxAmp/np.sum(myTxAmp)) 
                    + np.dot(mydRxDly, myRxAmp/np.sum(myRxAmp)))
        
        """ Define the output array """
        freq_delay = np.zeros(len(freq), dtype=np.complex128)
        d_freq = cuda.to_device(freq)
        d_freq_delay = cuda.to_device(freq_delay)
        d_myTxAmp = cuda.to_device(myTxAmp)
        d_myTxDly = cuda.to_device(myTxDly)
        d_myRxAmp = cuda.to_device(myRxAmp)
        d_myRxDly = cuda.to_device(myRxDly)
        
        threadsperblock = 32
        blockspergrid = (freq_delay.size + (threadsperblock-1))//threadsperblock
        freq_phase[blockspergrid, threadsperblock](d_freq, 
                                                   d_myTxAmp, 
                                                   d_myTxDly, 
                                                   d_myRxAmp, 
                                                   d_myRxDly, 
                                                   d_freq_delay)
        freq_delay = d_freq_delay.copy_to_host()
        
        dCorrection = np.exp(1j*2.0*np.pi*freq*pattern_delay)
        
        rate = 1j*np.pi*chirp_bandwidth/chirp_duration
        wcarrier = 2.0*1j*np.pi*chirp_carrier
        t = newFastTimes - targetRangeTime
        response = np.fft.ifft(np.fft.fft(np.exp( wcarrier*t + rate*(t-chirp_duration/2.0)**2 ))*freq_delay*dCorrection)
        response = response*np.exp( -wcarrier*(newFastTimes) )
        
        # Pulse compress the signal
        baseband_pulse = np.exp( rate*(np.arange(0,chirp_duration,dT) - chirp_duration/2.0)**2 )
        return eF*np.fft.ifft(np.conj(np.fft.fft(baseband_pulse, len(response)))*np.fft.fft(response))[0:nSamples]
    
    #%%
    from numba import cuda
    import math
    
    #@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], float64[:], complex128[:])', device=True)
    @cuda.jit()
    def freq_phase(freq, myTxAmp, myTxDly, myRxAmp, myRxDly, freq_delay):
        myTxLen = len(myTxAmp)
        myRxLen = len(myRxAmp)
        k = cuda.grid(1)
        w = -2.0*np.pi*freq[k]
        sTx = np.complex(0.0,0.0)
        for kk in range(myTxLen):
            sTx += myTxAmp[kk]*(math.cos(w*myTxDly[kk]) + 1j*math.sin(w*myTxDly[kk]))
            
        sRx = np.complex(0.0,0.0)
        for kk in range(myRxLen):
            sRx += myRxAmp[kk]*(math.cos(w*myRxDly[kk]) + 1j*math.sin(w*myRxDly[kk]))
            
        freq_delay[k] = sTx*sRx 

#@cuda.jit
#def freq_phase(freq, myTxAmp, myTxDly, myRxAmp, myRxDly, freq_delay):
#    start = cuda.grid(1)
#    stride = cuda.gridsize(1)
#    for k in range(start, freq.shape[0], stride):
#        w = 2.0*np.pi*freq[k]
#        sTx = np.complex(0.0,0.0)
#        for kk in range(myTxLen):
#            sTx += myTxAmp[kk]*(math.cos(-w*myTxDly[kk]) + 1j*math.sin(-w*myTxDly[kk]))
#            
#        sRx = np.complex(0.0,0.0)
#        for kk in range(myRxLen):
#            sRx += myRxAmp[kk]*(math.cos(-w*myRxDly[kk]) + 1j*math.sin(-w*myRxDly[kk]))
#            
#        freq_delay[k] = sTx*sRx 
    
#@cuda.jit
#def freq_phase(freq, myTxAmp, myTxDly, myRxAmp, myRxDly, freq_delay):
#    start = cuda.grid(1)
#    freq_delay[start] = 2.3 + 1.1j

#%%       
#t1 = time.time()
#freq_phase[24, 32](freq, myTxAmp, myTxDly, myRxAmp, myRxDly, freq_delay)
#t2 = time.time()
#print(t2 - t1)
#
##%%
#@cuda.jit(device=True)
#def polar_to_cartesian(rho, theta):
#    x = rho * math.cos(theta)
#    y = rho * math.sin(theta)
#    return x,y
#
#@vectorize(['float32(float32, float32, float32, float32)'], target='cuda')
#def polar_distance(rho1, theta1, rho2, theta2):
#    x1,y1 = polar_to_cartesian(rho1, theta1)
#    x2,y2 = polar_to_cartesian(rho2, theta2)
#    
#    return ((x1-x2)**2 + (y2 - y1)**2)**0.5