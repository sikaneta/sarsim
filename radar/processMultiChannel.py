# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:57:13 2019

@author: SIKANETAI
"""

import configuration.configuration as cfg
import argparse
import utils.fileio as fio

#%% Argparse stuff
parser = argparse.ArgumentParser(description="Multichannel process the data")

parser.add_argument("--config-xml", help="The config XML file", 
                    default = u'/home/ishuwa/simulation/40cm/40cm.xml')
parser.add_argument("--processed-file",
                    help="The name of the processed output file",
                    default = None)
parser.add_argument("--ridx",
                    help="""The range indeces to process. The process is
                            independent of the range variable so we can
                            easily split it into chunks""",
                    type=int,
                    nargs="+",
                    default=[0,None])
parser.add_argument("--xblock-size",
                    help="Size of the output data block in the X direction",
                    type=int,
                    default=None)

#%%
vv = parser.parse_args()

#%% Load the radar configuration
radar = cfg.loadConfiguration(vv.config_xml)

#%%
# output_file = "_".join(radar[0]['filename'].split("_")[0:-2] + 
#                        ["Xr%d" % vv.ridx[0], "mchanprocessed.npy"])
output_file = fio.fileStruct(radar[0]['filename'],
                             "mchan_processed",
                             "Xr%d" % vv.ridx[0],
                             "mchanprocessed.npy")

#%% Multi-channel process the data
procData, _ = cfg.multiChannelProcessMem(radar, 
                                         vv.ridx, 
                                         p=0.9)

#%% Write data to file
fio.writeSimFiles(output_file, 
                  procData, 
                  xblock=vv.xblock_size)