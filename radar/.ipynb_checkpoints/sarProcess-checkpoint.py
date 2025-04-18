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
procData = cfg.loadNumpy_mcp_data(vv.mchan_processed_file)

#%% Compute the ground point and associate slow time parameters
r_sys = cfg.radar_system(radar, bands)
r_sys.computeGroundPoint(radar, range_idx=400)

#%% Call the SAR processing algorithm
if vv.wk_processed_file is None:
    vv.wk_processed_file = "_".join(radar[0]['filename'].split("_")[0:-2] + 
                                       ["rX", "wkprocessed.npy"])
wkSignal = cfg.wkProcessNumba(procData, r_sys, os_factor=16, mem_cols = 16384,
                              tempFile=vv.wk_processed_file)

#%% Clear data so we can do the FFT with more memory
if vv.wk_processed_file is not None:
    del procData
    wkSignal = np.fft.ifft(np.load(vv.wk_processed_file), axis=1)
    wkSignal = wkSignal/np.max(np.abs(wkSignal))
    
#%%
mxcol = np.argmax(np.sum(np.abs(wkSignal), axis=0))
mxrow = np.argmax(np.sum(np.abs(wkSignal), axis=1))
print("Maximum for data located at row %d, col %d" % (mxrow, mxcol))

#%% Plot the processed signal
if vv.make_plots:
    DX = 200
    DY = 200
    plt.figure()
    plt.imshow(20.0*np.log10(np.abs(wkSignal[(mxrow-DY):(mxrow+DY), (mxcol-DX):(mxcol+DX)])))
    plt.clim(-50,00)
    plt.colorbar()
    plt.show()

#%% Plot cross-sections
if vv.make_plots:
    rows, cols = wkSignal.shape
    x = (np.arange(cols) - cols/2.0)/(r_sys.n_bands*r_sys.ksp)
    plt.figure()
    plt.plot(x, 20.0*np.log10(np.abs(wkSignal[mxrow,:])), x, 20.0*np.log10(np.abs(wkSignal[mxrow,:])), 'o')
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
    plt.plot(20.0*np.log10(np.abs(wkSignal[:,mxcol])))
    plt.title("Range cross-section")
    plt.xlabel('Range (sample)')
    plt.ylabel('Response (dB)')
    #plt.imshow(np.abs(wkSignal))
    #plt.clim(-100,10)
    plt.grid()
    plt.show()
