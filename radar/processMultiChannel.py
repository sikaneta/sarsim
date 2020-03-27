# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:57:13 2019

@author: SIKANETAI
"""

import configuration.configuration as cfg
import numpy as np
import argparse
import os

#%% Argparse stuff
parser = argparse.ArgumentParser(description="Multichannel process the data")

parser.add_argument("--config-xml", help="The config XML file", 
                    default = u'/home/ishuwa/simulation/40cm/simulation_40cm.xml')
parser.add_argument("--processed-file",
                    help="The name of the processed output file",
                    default = None)
parser.add_argument("--make-plots",
                    help="Generate plots along the way",
                    action='store_true',
                    default=True)
parser.add_argument("--target-range-index",
                    help="""The range sample index of the target.
                            This is needed to compute the reference phase
                            for the plot""",
                    type=int,
                    default=400)
vv = parser.parse_args()

#%% Load the radar configuration
radar = cfg.loadConfiguration(vv.config_xml)

#%%
output_file = "_".join(radar[0]['filename'].split("_")[0:-2] + ["Xr", "mchanprocessed.npy"])

#%% Define which bands to look at
bands = np.arange(-int(len(radar)/2)-2,int(len(radar)/2)+1+2)

#%% Multi-channel process the data
procData, _ = cfg.multiChannelProcess(radar, bands, p=0.9)

#%% Write data to file
np.save(output_file, procData)

#%% Plot as desired
if vv.make_plots:    
    import plot.simplot as sp
    # Define the folder in which to store plots
    sim_folder = os.path.join(os.path.split(vv.config_xml)[0], 
                              "simulation_plots")
    # Get an r_sys object
    r_sys = cfg.radar_system(radar, bands)
    
    # Compute the ground point and associate slow time parameters
    r_sys.computeGroundPoint(radar, range_idx=vv.target_range_index)

    sp.mchanPlot(procData, 
                 r_sys, 
                 interactive=False, 
                 folder=sim_folder, 
                 s_off=0.0)