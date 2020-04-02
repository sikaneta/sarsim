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
          Plot some figures from the multichannel processed
          data. These can be used to check the quality of the
          processing
          """
parser = argparse.ArgumentParser(description=purpose)

parser.add_argument("--config-xml", help="The config XML file", 
                    default = u'/home/ishuwa/simulation/40cm/simulation_40cm.xml')
parser.add_argument("--target-range-index",
                    help="""The range sample index of the target.
                            This is needed to compute the reference phase
                            for the last phase plot""",
                    type=int,
                    default=400)
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
if 'procData'not in locals():
    proc_file = fio.fileStruct(radar[0]['filename'],
                                        "mchan_processed",
                                        "Xr",
                                        "mchanprocessed.npy")
    print("Loading data from file: %s" % proc_file)
    procData = fio.loadSimFiles(proc_file, xidx=vv.xidx, ridx=vv.ridx)

#%% Define which bands to look at
#bands = np.arange(-int(len(radar)/2)-2,int(len(radar)/2)+1+2)

#%% Get an r_sys object
r_sys = cfg.loadRsys(radar)

#%% Compute the ground point and associate slow time parameters
r_sys.computeGroundPoint(radar, range_idx=vv.target_range_index)

#%% Define the folder in which to store plots
sim_folder = os.path.join(os.path.split(vv.config_xml)[0], 
                          "simulation_plots")

#%% Do the plotting
sp.mchanPlot(procData, 
             r_sys, 
             vv.interactive, 
             sim_folder, 
             vv.arclength_offset)