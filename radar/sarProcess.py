#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:02:31 2019

@author: ishuwa
"""

import configuration.configuration as cfg
import argparse
import utils.fileio as fio
import numpy as np

#%% Argparse stuff
parser = argparse.ArgumentParser(description="SAR process data that has been multi-channel processed")

parser.add_argument("--config-xml", 
                    help="The config XML file", 
                    required=True)
parser.add_argument("--mchan-processed-file",
                    help="The name of the multi-channel processed output file",
                    default = None)
parser.add_argument("--wk-processed-file",
                    help="The name of the w-k processed output file",
                    default = None)
parser.add_argument("--vanilla-stolt",
                    help="Apply vanilla Stolt interpolation a3=a4=0",
                    action="store_true")
parser.add_argument("--xidx",
                    help="""The azimuth indeces to process. The process can
                            proceed with the azimuth direction split into 
                            tiles and processed independently. For a given
                            range of azimuth samples, all range samples need
                            to be loaded into memory""",
                    type=int,
                    nargs="+",
                    default=[0,None])
parser.add_argument("--no-phase-correct",
                    help="""Do not Correct for difference between diffgeo and real""",
                    action="store_true")
parser.add_argument("--rblock-size",
                    help="Size of the output data block in the r direction",
                    type=int,
                    default=400)

#%%
vv = parser.parse_args()

#%% Load the radar configuration
radar = cfg.loadConfiguration(vv.config_xml)

#%% Load the data
if vv.mchan_processed_file is None:
    vv.mchan_processed_file = fio.fileStruct(radar[0]['filename'],
                                             "mchan_processed",
                                             "Xr",
                                             "mchanprocessed.npy")
    
#%% Compute the ground point and associate slow time parameters
r_sys = cfg.loadRsys(radar)
if vv.vanilla_stolt:
    print("Applying vanilla Stolt interpolation")
    r_sys.C.a3 = 0.0
    r_sys.C.a4 = 0.0
    
#%% Load the data
procData = fio.loadSimFiles(vv.mchan_processed_file, xidx=vv.xidx)
if not vv.no_phase_correct:
    print("Removing differential geo phase error")
    phsCorr = np.exp(-1j*r_sys.ks_phase_correction[vv.xidx[0]:vv.xidx[-1]])
    (rows,cols) = procData.shape
    for row in range(rows):
        procData[row,:] *= phsCorr[row]
        
#%% Do the SAR processing for this chunk
print("Processing")
print(vv.xidx)
wkSignal = cfg.wkProcessNumba(procData, 
                              r_sys, 
                              ks = r_sys.ks_full[vv.xidx[0]:vv.xidx[1]],
                              os_factor=16, 
                              mem_cols = np.min([cols, vv.rblock_size]),
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
