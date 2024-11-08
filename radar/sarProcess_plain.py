#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:02:31 2019

@author: ishuwa
"""

import configuration.configuration as cfg
import numpy as np
import matplotlib.pyplot as plt
import argparse

#%% Argparse stuff
parser = argparse.ArgumentParser(description="SAR process data that has been multi-channel processed")

parser.add_argument("--config-xml", help="The config XML file", 
                    default = u'/home/ishuwa/simulation/40cm/40cm_simulation.xml')
parser.add_argument("--mchan-processed-file",
                    help="The name of the multi-channel processed output file",
                    default = None)
parser.add_argument("--make-plots",
                    help="Generate plots along the way",
                    action="store_true",
                    default=False)
parser.add_argument("--wk-processed-file",
                    help="Temporary file in which to write the W-K data. This is needed for large files",
                    default=None)
parser.add_argument("--prf-factor",
                    help="""The integer increase in the PRF from the individual channel PRFs
                    Minimally, the number of beams times the number of channels, but
                    equivalent to the length of the bands array in processMultiChannel,
                    which is given by, as a default: 
                    len(np.arange(-int(len(radar)/2)-2,int(len(radar)/2)+3))
                    with a buffer of 2 bands on either side of the main""",
                    type=int,
                    default = None)
vv = parser.parse_args()

#%% Load the radar configuration
radar = cfg.loadConfiguration(vv.config_xml)

#%% Define which bands to look at
bands = np.arange(-int(len(radar)/2)-2,int(len(radar)/2)+1+2)

#%% Load the data
if vv.mchan_processed_file is None:
    vv.mchan_processed_file = "_".join(radar[0]['filename'].split("_")[0:-2] + 
                                       ["Xr", "mchanprocessed.npy"])

#%%
#procData = cfg.loadNumpy_mcp_data(vv.mchan_processed_file)
procData = np.load(vv.mchan_processed_file)

#%% Compute the ground point and associate slow time parameters
r_sys = cfg.radar_system(radar, bands)
r_sys.computeGroundPoint(radar, range_idx=400)

#%% Call the SAR processing algorithm
if vv.wk_processed_file is None:
    vv.wk_processed_file = "_".join(radar[0]['filename'].split("_")[0:-2] + 
                                       ["rX", "wkprocessed.npy"])

#%%
wkSignal = cfg.wkProcessNumba(procData, r_sys, os_factor=16, mem_rows = 8192,
                              tempFile=vv.wk_processed_file)

#%% Clear data so we can do the FFT with more memory
if vv.wk_processed_file is not None:
    try:
        del procData
    except NameError as nE:
        print("Mchan data already deleted")
    wkSignal = np.fft.ifft(np.load(vv.wk_processed_file), axis=0)
    wkSignal = wkSignal/np.max(np.abs(wkSignal))
    
#%%
mxcol = np.argmax(np.sum(np.abs(wkSignal), axis=0))
mxrow = np.argmax(np.sum(np.abs(wkSignal), axis=1))
print("Maximum for data located at row %d, col %d" % (mxrow, mxcol))

#%% Plot the processed signal
if vv.make_plots:
def makePlot(DX=100, DY=100):
    plt.figure()
    plt.imshow(20.0*np.log10(np.abs(wkSignal[(mxrow-DY):(mxrow+DY), (mxcol-DX):(mxcol+DX)])))
    plt.clim(-50,0)
    plt.colorbar()
    plt.title("Response (dB)")
    plt.xlabel("Range sample")
    plt.ylabel("Azimuth sample")
    plt.show()

#%% Plot cross-sections
if vv.make_plots:
def makeAziPlot(x0,x1):
    rows, cols = wkSignal.shape
    x = (np.arange(rows) - rows/2.0)/(r_sys.n_bands*r_sys.ksp) - 12.852 + 0.62
    xidx0 = np.argwhere(x>x0)[0][0]
    xidx1 = np.argwhere(x>x1)[0][0]
    plt.figure()
    plt.plot(x[xidx0:xidx1], 20.0*np.log10(np.abs(wkSignal[xidx0:xidx1,mxcol])), 
             x[xidx0:xidx1], 20.0*np.log10(np.abs(wkSignal[xidx0:xidx1,mxcol])), 'o')
    #plt.imshow(np.abs(wkSignal))
    #plt.clim(-100,10)
    plt.title("Azimuth cross-section")
    plt.xlabel('Azimuth (m)')
    plt.ylabel('Response (dB)')
    plt.grid()
    plt.show()

#%% Plot cross-sections
if vv.make_plots:
plt.figure()
plt.plot(20.0*np.log10(np.abs(wkSignal[mxrow,:])))
plt.plot(20.0*np.log10(np.abs(wkSignal[mxrow,:])),'.')
plt.title("Range cross-section")
plt.xlabel('Range (sample)')
plt.ylabel('Response (dB)')
#plt.imshow(np.abs(wkSignal))
#plt.clim(-100,10)
plt.grid()
plt.show()

#%%
if vv.make_plots:
    dSig = np.fft.fft(wkSignal[:,mxcol])

#%%
if vv.make_plots:
s = np.arange(r_sys.Na*r_sys.n_bands)/(r_sys.ksp*r_sys.n_bands)
s -= np.mean(s)
#%%
if vv.make_plots:
plt.figure()
#plt.plot(np.unwrap(np.angle(np.fft.fftshift(dSig*np.exp(1j*r_sys.ks_full*(np.min(s)+10.1))))))
plt.plot(r_sys.ks_full, np.abs(dSig*np.exp(1j*r_sys.ks_full*(s+10.0))),'.')
plt.title("Arclength wavenumber response")
plt.xlabel("Arclength wavenumber (rad/m)")
plt.ylable("Response")
plt.grid()
plt.show()
    
#%%
if vv.make_plots:
def plotAngle(s_off):
    plt.figure()
    plt.plot(sorted(r_sys.ks_full), np.unwrap(np.angle(np.fft.fftshift(dSig*np.exp(1j*r_sys.ks_full*(np.min(s)+s_off))))))
    #plt.plot(np.abs(np.fft.fftshift(dSig*np.exp(1j*r_sys.ks_full*(s+10.0)))))
    plt.title("Arclength wavenumber angle")
    plt.xlabel("Arclength wavenumber (rad/m)")
    plt.ylabel("Angle (rad)")
    plt.grid()
    plt.show()
    
    