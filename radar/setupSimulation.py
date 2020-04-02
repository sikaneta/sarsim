# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:11:41 2019

@author: SIKANETAI
"""

import configuration.configuration as cfg
import numpy as np
import argparse
    
#%% Load the data
desc = """Setup the simulation parameters
          as pickled objects. These objects include:
          - the radar object (with computed state vectors),
          - the radar system object (r_sys) which contains
          frequency arrays and the arclength object as well
          as the reference range to be used for the WK algorithm
          - the processing filter (H matrix)"""
parser = argparse.ArgumentParser(description=desc)

parser.add_argument("--config-xml", help="The config XML file", 
                    default = u'/home/ishuwa/simulation/40cm/simulation_40cm.xml')
parser.add_argument("--ref-range-idx",
                    help="""The reference range used for processing
                            If not specified, will use the center range""",
                    type = int,
                    default = None)
vv = parser.parse_args()

#%%
radar = cfg.loadConfiguration(vv.config_xml)

#%% Define which bands to reconstruct
bands = np.arange(-int(len(radar)/2)-2,int(len(radar)/2)+1+2)

#%% Generate the r_sys object
r_sys = cfg.computeStoreRsys(radar, bands, vv.ref_range_idx)

#%% Generate the processing filter
H, _ = cfg.computeStoreMultiProcFilter(radar) 
