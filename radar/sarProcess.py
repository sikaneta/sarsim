#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:02:31 2019

@author: ishuwa
"""

import configuration.configuration as cfg
import numpy as np
import argparse
import os
import utils.fileio as fio

#%% Argparse stuff
parser = argparse.ArgumentParser(description="SAR process data that has been multi-channel processed")

parser.add_argument("--config-xml", help="The config XML file", 
                    default = u'/home/ishuwa/simulation/40cm/simulation_40cm.xml')
parser.add_argument("--mchan-processed-file",
                    help="The name of the multi-channel processed output file",
                    default = None)
parser.add_argument("--wk-processed-file",
                    help="The name of the w-k processed output file",
                    default = None)
# parser.add_argument("--make-plots",
#                     help="Generate plots along the way",
#                     action="store_true",
#                     default=True)
parser.add_argument("--xidx",
                    help="""The azimuth indeces to process. The process can
                            proceed with the azimuth direction split into 
                            tiles and processed independently. For a given
                            range of azimuth samples, all range samples need
                            to be loaded into memory""",
                    type=int,
                    nargs="+",
                    default=[0,None])
# parser.add_argument("--prf-factor",
#                     help="""The integer increase in the PRF from the individual channel PRFs
#                     Minimally, the number of beams times the number of channels, but
#                     equivalent to the length of the bands array in processMultiChannel,
#                     which is given by, as a default: 
#                     len(np.arange(-int(len(radar)/2)-2,int(len(radar)/2)+3))
#                     with a buffer of 2 bands on either side of the main""",
#                     type=int,
#                     default = None)
# parser.add_argument("--target-range-index",
#                     help="""The reference range index.
#                             This is the index of the range used for the w-k
#                             algorithm""",
#                     type=int,
#                     default=None)
parser.add_argument("--rblock-size",
                    help="Size of the output data block in the r direction",
                    type=int,
                    default=400)
vv = parser.parse_args()

#%% Load the radar configuration
radar = cfg.loadConfiguration(vv.config_xml)

#%% Load the data
if vv.mchan_processed_file is None:
    # vv.mchan_processed_file = "_".join(radar[0]['filename'].split("_")[0:-2] + 
    #                                    ["Xr", "mchanprocessed.npy"])
    # head, tail = os.path.split(vv.mchan_processed_file)
    # vv.mchan_processed_file = os.path.join(head,"mchan_proc",tail)
    vv.mchan_processed_file = fio.fileStruct(radar[0]['filename'],
                                             "mchan_processed",
                                             "Xr",
                                             "mchanprocessed.npy")
    
#%%
procData = fio.loadSimFiles(vv.mchan_processed_file, xidx=vv.xidx)

#%% Compute the ground point and associate slow time parameters
r_sys = cfg.loadRsys(radar)
#r_sys.computeGroundPoint(radar, range_idx=vv.target_range_index)

#%% Do the SAR processing for this chunk
ks = r_sys.ks_full[vv.xidx[0]:vv.xidx[1]]
wkSignal = cfg.wkProcessNumba(procData, 
                              r_sys, 
                              ks = ks,
                              os_factor=16, 
                              mem_rows = 8192)

#%% Call the SAR processing algorithm
if vv.wk_processed_file is None:
    # vv.wk_processed_file = "_".join(radar[0]['filename'].split("_")[0:-2] + 
    #                                    ["X%dr" % vv.xidx[0], "wkprocessed.npy"])
    # head, tail = os.path.split(vv.wk_processed_file)
    # vv.wk_processed_file = os.path.join(head,"wk_proc",tail)
    vv.wk_processed_file = fio.fileStruct(radar[0]['filename'],
                                         "wk_processed",
                                         "X%dr" % vv.xidx[0],
                                         "wkprocessed.npy")
    
#%% Write the data to file
fio.writeSimFiles(vv.wk_processed_file, 
                  wkSignal, 
                  rblock=vv.rblock_size)