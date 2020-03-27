# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:57:13 2019

@author: SIKANETAI
"""

import configuration.configuration as cfg
import numpy as np
import argparse

#%% Argparse stuff
parser = argparse.ArgumentParser(description="Multichannel process the data")

parser.add_argument("--config-xml", help="The config XML file", 
                    default = u'/home/ishuwa/simulation/40cm/simulation_40cm.xml')
parser.add_argument("--processed-file",
                    help="The name of the processed output file",
                    default = None)
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

