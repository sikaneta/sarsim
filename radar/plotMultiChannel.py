#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:38:54 2020

@author: ishuwa
"""

import configuration.configuration as cfg
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

#%% Argparse stuff
purpose = """
          Plot some figures from the multichannel processed
          data. These can be used to check the quality of the
          processing
          """
parser = argparse.ArgumentParser(description=purpose)

parser.add_argument("--config-xml", help="The config XML file", 
                    default = u'/home/ishuwa/simulation/40cm/simulation_40cm.xml')
parser.add_argument("--target-range-index",
                    help="Define the index of the target in range",
                    type=int,
                    default=400)
parser.add_argument("--interactive",
                    help="Interactive mode. Program halts until figures are closed",
                    action="store_true",
                    default=False)
            
vv = parser.parse_args()

#%% Load the radar configuration
if 'radar' not in locals():
    radar = cfg.loadConfiguration(vv.config_xml)

#%% Load the data
if 'procData'not in locals():
    proc_file = "_".join(radar[0]['filename'].split("_")[0:-2] + 
                       ["Xr", "mchanprocessed.npy"])
    print("Loading data from file: %s" % proc_file)
    procData = np.load(proc_file)

#%% Define which bands to look at
bands = np.arange(-int(len(radar)/2)-2,int(len(radar)/2)+1+2)

#%% Get an r_sys object
r_sys = cfg.radar_system(radar, bands)

#%% Compute the ground point and associate slow time parameters
r_sys.computeGroundPoint(radar, range_idx=vv.target_range_index)

#%% Compute the slow time parameter
s = np.arange(r_sys.Na*r_sys.n_bands)/(r_sys.ksp*r_sys.n_bands)
s = s - np.mean(s)

#%% Define the folder in which to store plots
sim_folder = os.path.join(os.path.split(vv.config_xml)[0], 
                          "simulation_plots")

#%% Plot data in the Doppler domain
flatProc = np.sum(procData, axis=1)
print("Plotting the Doppler domain signal amplitude")
plt.figure()
plt.plot(sorted(r_sys.ks_full), np.abs(flatProc)[r_sys.ks_full_idx],'.')
plt.title("Computed antenna pattern")
plt.xlabel("Azimuth wavenumber (m$^{-1}$)")
plt.ylabel("Gain (Natural units)")
plt.grid()
if vv.interactive:
    plt.show()
else:
    plt.savefig(os.path.join(sim_folder, "mchan_doppler_amplitude.png"),
            transparent=True)
    plt.close()

#%% Sum across rows to get an azimuth signal
print("Computing the time domain signal")
flatProc = np.fft.ifft(flatProc)
    
#%% Plot the time domain signal across azimuth
print("Plotting the time domain signal amplitude")
plt.figure()
plt.grid()
plt.plot(s, np.abs(flatProc))
plt.title("Spatial domain reconstructed signal")
plt.xlabel("Azimuth (arclength m)")
plt.ylabel("Response")
if vv.interactive:
    plt.show()
else:
    plt.savefig(os.path.join(sim_folder, "mchan_arclength_amplitude.png"),
            transparent=True)
    plt.close()

#%% Compute the range curve
cdf = r_sys.C.cdf
rngs_curve = np.outer(r_sys.C.R, s**0) + np.outer(cdf[1],s) + np.outer(cdf[2], (s**2)/2.0) + np.outer(cdf[3], (s**3)/6.0)
rngs = np.sqrt(np.sum(rngs_curve*rngs_curve, axis=0))

#%% Create the inverse phase function
rC = np.exp(-1j*r_sys.kr[0]*rngs)

#%% Plot the phase compensated signal in the time domain
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
if vv.interactive:
    plt.show()
else:
    plt.savefig(os.path.join(sim_folder, "mchan_arclength_phase.png"),
            transparent=True)
    plt.close()