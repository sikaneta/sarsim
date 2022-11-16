# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:11:41 2019

@author: SIKANETAI
"""

import configuration.configuration as cfg
import argparse
    
#%% Load the data
parser = argparse.ArgumentParser(description="Generate simulated SAR data")

parser.add_argument("--config-xml", 
                    help="The config XML file", 
                    required=True)
parser.add_argument("--channel-idxs",
                    help=""" The indexes of the channels to create. This 
                    allows the program to be run simulataneously on different 
                    CPUs. Each channel is of the form cXbY for channel X, 
                    beam Y""",
                    nargs = "*",
                    type=int,
                    default = None)
parser.add_argument("--target-rangeidx",
                    help="The range index of the target to simulate",
                    type=int,
                    default=400)
parser.add_argument("--rblock-size",
                    help="""The output files for each beam/channel will be
                            written to files with the range dimension of
                            size range block. 
                            i.e. with 
                            - a range block size of --range-block=512 
                            - a data file with name (...rX_c0b0.npy)
                            - a data size of 1200X16384 in range by
                              azimuth respectively (rX) 
                            then will write several files of names 
                            ...r0X0_c0b0.npy    (has 512 rows)
                            ...r512X0_c0b0.npy  (has 512 rows)
                            ...r1024X0_c0b0.npy (has 1200-1024 rows)
                            If not specified, will set block to the total
                            number of range samples.""",
                    type=int,
                    default = None)

#%%
vv = parser.parse_args()

#%%
radar = cfg.loadConfiguration(vv.config_xml)

#%% Generate a reference ground point
pointXYZ, satSV, satTime = cfg.computeReferenceGroundPoint(radar, 
                                                      None, 
                                                      vv.target_rangeidx, 
                                                      None)  
satXYZ = satSV[0:3]
satvXvYvZ = satSV[3:]  
       

#%% Generate the data only for the selected channels
if vv.channel_idxs is None:
    vv.channel_idxs = range(len(radar))
    
cfg.computeSignal([radar[k] for k in vv.channel_idxs], 
                  pointXYZ, 
                  satXYZ,
                  vv.rblock_size)