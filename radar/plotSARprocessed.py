#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:38:54 2020

@author: ishuwa
"""

import configuration.configuration as cfg
import plot.simplot as sp
import numpy as np
import argparse
import os
import utils.fileio as fio
from tqdm import tqdm

#%% Argparse stuff
purpose = """
          Plot some figures from the SAR processed
          data. These can be used to check the quality of the
          processing
          """
parser = argparse.ArgumentParser(description=purpose)

parser.add_argument("--config-xml", help="The config XML file", 
                    default = u'/home/ishuwa/simulation/40cm/simulation_40cm.xml')

parser.add_argument("--arclength-offset",
                    help="Offset of the target from zero in arclength (m)",
                    type=float,
                    default=0.0)
parser.add_argument("--ridx",
                    help="The range indeces to examine",
                    type=int,
                    nargs=2,
                    default=[0,None])
parser.add_argument("--xidx",
                    help="The azimuth indeces to examine",
                    type=int,
                    nargs=2,
                    default=[0,None])
parser.add_argument("--interactive",
                    help="Interactive mode. Program halts until figures are closed",
                    action="store_true",
                    default=False)
            
vv = parser.parse_args()

#%% Load the radar configuration
if 'radar' not in locals():
    radar = cfg.loadConfiguration(vv.config_xml)
    
#%% Get an r_sys object
r_sys = cfg.loadRsys(radar)

#%% Load the data
if 'wkSignal'not in locals():
    proc_file = fio.fileStruct(radar[0]['filename'],
                                        "wk_processed",
                                        "Xr",
                                        "wkprocessed.npy")
    print("Loading data from file: %s" % proc_file)
    wkSignal = fio.loadSimFiles(proc_file, xidx=vv.xidx, ridx=vv.ridx)
    
    # Shift the signal as required
    print("Attempting to shift the signal...")
    s = np.arange(r_sys.Na*r_sys.n_bands)/(r_sys.ksp*r_sys.n_bands)
    s -= np.mean(s)
    mxcol = np.argmax(wkSignal[0,:])
    intf = wkSignal[0:-1, mxcol]*np.conj(wkSignal[1:, mxcol])
    dks = r_sys.ks_full[1] - r_sys.ks_full[0]
    c_ang = np.angle(np.sum(intf)) - dks*np.min(s)
    s_off = c_ang/dks
    p_fct = np.exp(1j*r_sys.ks_full*s_off)
    rows, cols = wkSignal.shape
    for k in tqdm(np.arange(cols)):
        wkSignal[:, k] *= p_fct
    print("Computing the FFT of the signal ...")
    wkSignal = np.fft.ifft(wkSignal, axis=0)
    wkSignal = wkSignal/np.max(np.abs(wkSignal))



#%% Define the folder in which to store plots
sim_folder = os.path.join(os.path.split(vv.config_xml)[0], 
                          "simulation_plots")

#%% Do the plotting
sp.sarprocPlot(wkSignal, 
               r_sys, 
               interactive=vv.interactive, 
               folder=sim_folder, 
               s_off=vv.arclength_offset)