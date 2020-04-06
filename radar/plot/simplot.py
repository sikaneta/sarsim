#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:43:14 2020

@author: ishuwa
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from common.utils import upsampleSignal

#%% Define a function to do the mchan plots
def mchanPlot(procData, 
              r_sys, 
              interactive=False, 
              folder=".", 
              s_off = 0.0):
    # Plot data in the Doppler domain
    flatProc = np.sum(procData, axis=1)
    print("Plotting the Doppler domain signal amplitude")
    plt.figure()
    plt.plot(sorted(r_sys.ks_full), np.abs(flatProc)[r_sys.ks_full_idx],'.')
    plt.title("Computed antenna pattern")
    plt.xlabel("Azimuth wavenumber (m$^{-1}$)")
    plt.ylabel("Gain (Natural units)")
    plt.grid()
    if interactive:
        plt.show()
    else:
        plt.savefig(os.path.join(folder, "mchan_doppler_amplitude.png"),
                transparent=True)
        plt.close()
    
    # Sum across rows to get an azimuth signal
    print("Computing the time domain signal")
    flatProc = np.fft.ifft(flatProc)
        
    # Plot the time domain signal across azimuth
    s = np.arange(r_sys.Na*r_sys.n_bands)/(r_sys.ksp*r_sys.n_bands)
    s = s - np.mean(s) + s_off
    print("Plotting the time domain signal amplitude")
    plt.figure()
    plt.grid()
    plt.plot(s, np.abs(flatProc))
    plt.title("Spatial domain reconstructed signal")
    plt.xlabel("Azimuth (arclength m)")
    plt.ylabel("Response")
    if interactive:
        plt.show()
    else:
        plt.savefig(os.path.join(folder, "mchan_arclength_amplitude.png"),
                transparent=True)
        plt.close()
    
    # Compute the range curve
    cdf = r_sys.C.cdf
    rngs_curve = np.outer(r_sys.C.R, s**0) + np.outer(cdf[1],s) + np.outer(cdf[2], (s**2)/2.0) + np.outer(cdf[3], (s**3)/6.0)
    rngs = np.sqrt(np.sum(rngs_curve*rngs_curve, axis=0))
    
    # Create the inverse phase function
    rC = np.exp(-1j*r_sys.kr[0]*rngs)
    
    # Plot the phase compensated signal in the time domain
    print("Plotting the time domain signal corrected phase")
    minsidx = np.argmin(np.abs(s))
    unwrpangle = np.unwrap(np.angle(flatProc*np.conj(rC)))
    plt.figure()
    plt.plot(s, unwrpangle,'.')
    plt.title("Unwrapped angle of signal multiplied by inverse of phase")
    plt.xlabel("Azimuth (arclength m)")
    plt.ylabel("Phase (rad)")
    plt.ylim([unwrpangle[minsidx] - 100, unwrpangle[minsidx] + 100])
    plt.grid()
    if interactive:
        plt.show()
    else:
        plt.savefig(os.path.join(folder, "mchan_arclength_phase.png"),
                transparent=True)
        plt.close()
        
def sarprocPlot(wkSignal,
                r_sys,
                interactive=False, 
                folder=".",
                s_off = 5.61):
    
    # Calculate the maximum
    rows, cols = wkSignal.shape
    mxcol = np.argmax(np.sum(np.abs(wkSignal), axis=0))
    mxrow = np.argmax(np.sum(np.abs(wkSignal), axis=1))
    print("Maximum for data located at row %d, col %d" % (mxrow, mxcol))
    
    # Calculate the arclength parameters
    s = np.arange(r_sys.Na*r_sys.n_bands)/(r_sys.ksp*r_sys.n_bands)
    s -= np.mean(s)

    # Calculate the FFT of the signal at the maximum to examine the Doppler
    # response
    dSig = np.fft.fft(wkSignal[:,mxcol])*np.exp(1j*r_sys.ks_full*(np.min(s)))
    
    y = wkSignal[:,mxcol]
    
    # Plot the data
    for k in range(3,10):
        makePlot(wkSignal, 
                 mxrow, 
                 mxcol, 
                 DX=2**k, 
                 DY=2**k, 
                 folder=folder,
                 interactive=interactive)
    for k in range(3,10):
        makeAziPlot(wkSignal, 
                    mxrow, 
                    mxcol, 
                    s, 
                    DY=2**k, 
                    folder = folder,
                    interactive=interactive)
    for k in range(3,6):
        makeRngPlot(wkSignal, 
                    mxrow, 
                    mxcol, 
                    r_sys.r-r_sys.r[mxcol], 
                    DX=2**k, 
                    folder = folder,
                    interactive=interactive)
    for k in range(3,10):
        makeOversampledPlot(wkSignal[:,mxcol],
                            mxrow,
                            s[1]-s[0],
                            D = 2**k,
                            os_factor = 8,
                            folder = folder,
                            interactive = interactive,
                            xlabel = "Azimuth (arclength m)",
                            title = "Response (dB)",
                            filename = "wk_response_s_os_%d.png" % (2**k))
    plotAngle(dSig, 
              r_sys, 
              folder = folder,
              interactive=interactive)
    
    plotKS(dSig, 
           r_sys, 
           folder = folder,
           interactive=interactive)

#%% Function to plot the processed signal
def makePlot(wkSignal, 
             mxrow, 
             mxcol, 
             DX=100, 
             DY=100, 
             cmin=-50,
             folder = ".",
             interactive = False):
    rows, cols = wkSignal.shape
    Y0 = np.max([0, mxrow-DY])
    Y1 = np.min([rows, mxrow+DY])
    X0 = np.max([0, mxcol-DX])
    X1 = np.min([cols, mxcol+DX])
    plt.figure()
    plt.imshow(20.0*np.log10(np.abs(wkSignal[Y0:Y1,X0:X1])))
    plt.clim(cmin,0)
    plt.colorbar()
    plt.title("Response (dB)")
    plt.xlabel("Range sample")
    plt.ylabel("Azimuth sample")
    if interactive:
        plt.show()
    else:
        plt.savefig(os.path.join(folder, "wk_response_%dx%d.png" % (DX,DY)),
                    transparent=True)
        plt.close()

#%% Make oversampled plot
def makeOversampledPlot(signal,
                        mxidx,
                        ds,
                        D = 100,
                        os_factor = 8,
                        folder = ".",
                        interactive = False,
                        xlabel = "",
                        title = "",
                        filename = "plot.png"):
    Y0 = np.max([0, mxidx-D])
    Y1 = np.min([len(signal), mxidx+D])
    usignal = upsampleSignal(signal[Y0:Y1], os_factor, 0)
    x = np.arange(len(usignal))*ds/os_factor
    x -= x[np.argmax(np.abs(usignal))]
    plt.figure()
    plt.plot(x, 20.0*np.log10(np.abs(usignal)),
             x, 20.0*np.log10(np.abs(usignal)), '.')
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    if interactive:
        plt.show()
    else:
        plt.savefig(os.path.join(folder, filename),
                    transparent=True)
        plt.close()
        
#%% Function to make azimuth plot
def makeAziPlot(wkSignal, 
                mxrow, 
                mxcol, 
                s,
                DY=100, 
                folder = ".",
                interactive = False):
    rows, cols = wkSignal.shape
    Y0 = np.max([0, mxrow-DY])
    Y1 = np.min([rows, mxrow+DY])
    plt.figure()
    plt.plot(s[Y0:Y1], 
             20.0*np.log10(np.abs(wkSignal[Y0:Y1,mxcol])),
             s[Y0:Y1], 
             20.0*np.log10(np.abs(wkSignal[Y0:Y1,mxcol])),
             '.')
    plt.grid()
    plt.title("Response (dB)")
    plt.xlabel("Azimuth (arclength m)")
    if interactive:
        plt.show()
    else:
        plt.savefig(os.path.join(folder, "wk_response_arclength_%d.png" % DY),
                    transparent=True)
        plt.close()

#%% Function to make azimuth plot
def makeRngPlot(wkSignal, 
                mxrow, 
                mxcol, 
                r,
                DX=100, 
                folder = ".",
                interactive = False,
                ylim = [-30,0]):
    rows, cols = wkSignal.shape
    X0 = np.max([0, mxcol-DX])
    X1 = np.min([cols, mxcol+DX])
    plt.figure()
    plt.plot(r[X0:X1], 
             20.0*np.log10(np.abs(wkSignal[mxrow, X0:X1])),
             r[X0:X1], 
             20.0*np.log10(np.abs(wkSignal[mxrow, X0:X1])),
             '.')
    plt.ylim(ylim)
    plt.title("Response (dB)")
    plt.xlabel("Slant range (m)")
    if interactive:
        plt.show()
    else:
        plt.savefig(os.path.join(folder, "wk_response_srange_%d.png" % DX),
                    transparent=True)
        plt.close() 
        
    
#%% Plot the phase of the processed signal
def plotAngle(signal,
              r_sys,
              D = 10,
              interactive = False,
              folder = "."):
    
    unwrpangle = np.unwrap(np.angle(np.fft.fftshift(signal)))
    minsidx = int(unwrpangle.shape[0]/2)
    plt.figure()
    plt.plot(sorted(r_sys.ks_full), 
             unwrpangle,'.')
    plt.ylim([unwrpangle[minsidx] - D, unwrpangle[minsidx] + D])
    plt.title("Arclength wavenumber angle")
    plt.xlabel("Arclength wavenumber (rad/m)")
    plt.ylabel("Angle (rad)")
    plt.grid()
    if interactive:
        plt.show()
    else:
        plt.savefig(os.path.join(folder, "wk_doppler_response_angle.png"),
                    transparent=True)
        plt.close()

    
    
#%% Plot the amplitude of the processed signal
def plotKS(signal,
           r_sys,
           ylim=[-30,0],
           interactive = False,
           folder = "."):
    
    y = 20*np.log10(np.abs(np.fft.fftshift(signal)))
    y -= np.max(y)
    plt.figure()
    plt.plot(sorted(r_sys.ks_full), y,'.')
    plt.ylim(ylim)
    plt.title("Arclength wavenumber response")
    plt.xlabel("Arclength wavenumber (rad/m)")
    plt.ylabel("Response (amplitude)")
    plt.grid()
    if interactive:
        plt.show()
    else:
        plt.savefig(os.path.join(folder, "wk_doppler_response_amplitude.png"),
                    transparent=True)
        plt.close()
    
