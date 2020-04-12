#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:02:31 2019

@author: ishuwa
"""

import configuration.configuration as cfg
import argparse
import utils.fileio as fio

#%% Argparse stuff
parser = argparse.ArgumentParser(description="SAR process data that has been multi-channel processed")

parser.add_argument("--config-xml", help="The config XML file", 
                    default = u'/home/ishuwa/simulation/20cm/simulation_20cm.xml')
parser.add_argument("--mchan-processed-file",
                    help="The name of the multi-channel processed output file",
                    default = None)
parser.add_argument("--wk-processed-file",
                    help="The name of the w-k processed output file",
                    default = None)
parser.add_argument("--xidx",
                    help="""The azimuth indeces to process. The process can
                            proceed with the azimuth direction split into 
                            tiles and processed independently. For a given
                            range of azimuth samples, all range samples need
                            to be loaded into memory""",
                    type=int,
                    nargs="+",
                    default=[0,None])
parser.add_argument("--rblock-size",
                    help="Size of the output data block in the r direction",
                    type=int,
                    default=400)
vv = parser.parse_args()

#%% Load the radar configuration
radar = cfg.loadConfiguration(vv.config_xml)

#%% Load the data
if vv.mchan_processed_file is None:
    vv.mchan_processed_file = fio.fileStruct(radar[0]['filename'],
                                             "mchan_processed",
                                             "Xr",
                                             "mchanprocessed.npy")
    
#%% Load the data
procData = fio.loadSimFiles(vv.mchan_processed_file, xidx=vv.xidx)

#%% Compute the ground point and associate slow time parameters
r_sys = cfg.loadRsys(radar)

#%% Do the SAR processing for this chunk
print("Processing")
print(vv.xidx)
wkSignal = cfg.wkProcessNumba(procData, 
                              r_sys, 
                              ks = r_sys.ks_full[vv.xidx[0]:vv.xidx[1]],
                              os_factor=16, 
                              mem_rows = 8192)

#%% Call the SAR processing algorithm
if vv.wk_processed_file is None:
    vv.wk_processed_file = fio.fileStruct(radar[0]['filename'],
                                         "wk_processed",
                                         "X%dr" % vv.xidx[0],
                                         "wkprocessed.npy")
    
#%% Write the data to file
fio.writeSimFiles(vv.wk_processed_file, 
                  wkSignal, 
                  rblock=vv.rblock_size)