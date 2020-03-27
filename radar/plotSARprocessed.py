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
    proc_file = "_".join(radar[0]['filename'].split("_")[0:-2] + 
                         ["rX", "wkprocessed.npy"])
    print("Loading data from file: %s" % proc_file)
    wkSignal = np.fft.ifft(np.load(proc_file), axis=0)
    wkSignal = wkSignal/np.max(np.abs(wkSignal))

#%% Define which bands to look at
bands = np.arange(-int(len(radar)/2)-2,int(len(radar)/2)+1+2)

#%% Get an r_sys object
r_sys = cfg.radar_system(radar, bands)

#%% Define the folder in which to store plots
sim_folder = os.path.join(os.path.split(vv.config_xml)[0], 
                          "simulation_plots")

#%% Do the plotting
sp.sarprocPlot(wkSignal, 
               r_sys, 
               interactive=False, 
               folder=sim_folder, 
               s_off=vv.arclength_offset)