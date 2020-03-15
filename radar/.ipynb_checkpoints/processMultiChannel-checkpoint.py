# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:57:13 2019

@author: SIKANETAI
"""

import configuration.configuration as cfg
import numpy as np
import matplotlib.pyplot as plt
from measurement.measurement import state_vector
from measurement.arclength import slow
import argparse
from measurement.measurement import state_vector
import os
import argparse
#%matplotlib notebook

#%% Argparse stuff
parser = argparse.ArgumentParser(description="Multichannel process the data")

parser.add_argument("--config-xml", help="The config XML file", 
                    default = u'/home/ishuwa/simulation/40cm/40cm_simulation.xml')
parser.add_argument("--processed-file",
                    help="The name of the processed output file",
                    default = None)
parser.add_argument("--make-plots",
                    help="Generate plots along the way",
                    action='store_true',
                    default=False)
vv = parser.parse_args()

#%% Load the radar configuration
#radar = cfg.loadConfiguration("/home/ishuwa/simulation/40cm/reduced_40cm_simulation.xml")
radar = cfg.loadConfiguration(vv.config_xml)

#%%
output_file = "_".join(radar[0]['filename'].split("_")[0:-2] + ["Xr", "mchanprocessed.npy"])

#%% Define which bands to look at
bands = np.arange(-int(len(radar)/2)-2,int(len(radar)/2)+1+2)

#%% Get an r_sys object
r_sys = cfg.radar_system(radar, bands)

#%% Multi-channel process the data
procData, _ = cfg.multiChannelProcess(radar, bands, p=0.5)

#%% Look at a plot
if vv.make_plots:
    plt.figure()
    plt.plot(sorted(r_sys.ks_full), np.abs(np.sum(procData,axis=1))[r_sys.ks_full_idx],'.')
    #plt.plot(sorted(r_sys.ks_full), np.abs(np.sum(procData,axis=1)),'.')
    plt.title("Computed antenna pattern")
    plt.xlabel("Azimuth wavenumber (m$^{-1}$)")
    plt.ylabel("Gain (Natural units)")
    plt.grid()
    plt.show()

#%% Write data to file
np.save(output_file, procData)

#%% Transform the data into the time domain
procData = np.fft.ifft(procData, axis=0)
#%%
if vv.make_plots:
    flatProc = np.sum(procData, axis=1)
    
#%%
if vv.make_plots:
    plt.figure()
    plt.grid()
    plt.plot(np.abs(flatProc))
    plt.title("Spatial domain reconstructed signal")
    plt.xlabel("Sample number azimuth")
    plt.ylabel("Response")
    plt.show()

    # Compute the ground point and associate slow time parameters
    r_sys.computeGroundPoint(radar, range_idx=400)
    
    # Compute the slow time parameter
    s = np.arange(r_sys.Na*r_sys.n_bands)/(r_sys.ksp*r_sys.n_bands)
    s = s - np.mean(s) - 10
    cdf = r_sys.C.cdf
    
    # Compute the range curve
    rngs_curve = np.outer(r_sys.C.R, s**0) + np.outer(cdf[1],s) + np.outer(cdf[2], (s**2)/2.0) + np.outer(cdf[3], (s**3)/6.0)
    rngs = np.sqrt(np.sum(rngs_curve*rngs_curve, axis=0))
    
    # Create the inverse phase function
    rC = np.exp(-1j*r_sys.kr[0]*rngs)
    
    plt.figure()
    plt.plot(np.unwrap(np.angle(flatProc*np.conj(rC))),'.')
    plt.title("Unwrapped angle of signal multiplied by inverse of phase")
    plt.xlabel("Sample number")
    plt.ylabel("Phase (rad)")
    plt.grid()
    plt.show()