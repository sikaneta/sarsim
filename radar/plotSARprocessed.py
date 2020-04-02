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

#%% Load the data
if 'wkSignal'not in locals():
    proc_file = fio.fileStruct(radar[0]['filename'],
                                        "wk_processed",
                                        "Xr",
                                        "wkprocessed.npy")
    print("Loading data from file: %s" % proc_file)
    wkSignal = fio.loadSimFiles(proc_file, xidx=vv.xidx, ridx=vv.ridx)
    wkSignal = np.fft.ifft(wkSignal, axis=0)
    wkSignal = wkSignal/np.max(np.abs(wkSignal))

#%% Get an r_sys object
r_sys = cfg.loadRsys(radar)

#%% Define the folder in which to store plots
sim_folder = os.path.join(os.path.split(vv.config_xml)[0], 
                          "simulation_plots")

#%% Do the plotting
sp.sarprocPlot(wkSignal, 
               r_sys, 
               interactive=vv.interactive, 
               folder=sim_folder, 
               s_off=vv.arclength_offset)