# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:57:13 2019

@author: SIKANETAI
"""

import configuration.configuration as cfg
import numpy as np
import argparse

#%% Argparse stuff
parser = argparse.ArgumentParser(description="Generate the multichannel filter")

parser.add_argument("--config-xml", 
                    help="The config XML file", 
                    required=True)
vv = parser.parse_args()

#%% Load the radar configuration
radar = cfg.loadConfiguration(vv.config_xml)

#%% Define which bands to look at
bands = np.arange(-int(len(radar)/2)-2,int(len(radar)/2)+1+2)

#%% Multi-channel process the data
_,_ = cfg.computeStoreMultiProcFilter(radar, bands)
