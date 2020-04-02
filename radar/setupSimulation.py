# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:11:41 2019

@author: SIKANETAI
"""

import configuration.configuration as cfg
import numpy as np
import argparse
import os
    
#%% Load the data
desc = """Setup the simulation parameters
          as pickled objects. These objects include:
          - the radar object (with computed state vectors),
          - the radar system signal processing object (r_sys) which 
          contains frequency arrays and the arclength object as well
          as the reference range to be used for the WK algorithm
          - the processing filter (H matrix)"""
parser = argparse.ArgumentParser(description=desc)

parser.add_argument("--config-xml", help="The config XML file", 
                    default = u'/home/ishuwa/simulation/40cm/simulation_40cm.xml')
parser.add_argument("--recompute-rsys",
                    help="Recompute the signal processing object",
                    action="store_true",
                    default=False)
parser.add_argument("--recompute-Hfilter",
                    help="Recompute the multi channel filter",
                    action="store_true",
                    default=False)
parser.add_argument("--ref-range-idx",
                    help="""The reference range used for processing
                            If not specified, will use the center range""",
                    type = int,
                    default = None)
parser.add_argument("--target-rangeidx",
                    help="The range index of the target to simulate",
                    type=int,
                    default=400)
parser.add_argument("--rblock-size",
                    help="Size of data blocks in the r direction",
                    type=int,
                    default=400)
parser.add_argument("--xblock-size",
                    help="Size of data blocks in the X direction",
                    type=int,
                    default=16384)
vv = parser.parse_args()

#%% Make sure we're looking at the absolute path
vv.config_xml = os.path.abspath(vv.config_xml)

#%% Compute the array of radar objects
print("""Computing the radar system acquisition object""")
radar = cfg.loadConfiguration(vv.config_xml)

#%% Define which bands to reconstruct
bands = np.arange(-int(len(radar)/2)-2,int(len(radar)/2)+1+2)

#%% Generate the r_sys object
print("""Computing the radar system signal processing object""")
if vv.recompute_rsys:
    r_sys = cfg.computeStoreRsys(radar, bands, vv.ref_range_idx)
else:
    r_sys = cfg.loadRsys(radar) or cfg.computeStoreRsys(radar, 
                                                        bands, 
                                                        vv.ref_range_idx)

#%% Generate the processing filter
print("""Computing the multi channel processing filter""")
if vv.recompute_Hfilter:
    H,_ = cfg.computeStoreMultiProcFilter(radar)
else:
    H = cfg.loadMultiProcFilter(radar)
    if H is None:
        H,_ = cfg.computeStoreMultiProcFilter(radar)[0]

#%% Generate the list of commands that should be run with the given block sizes
all_commands = ["#!/bin/bash"]

#%% Generate commands to create the raw signal
gen_command = ["python -m generateMsar"]
for chan_num in range(len(radar)):
    gen_args = ["--config-xml %s" % vv.config_xml,
                "--channel-idxs %d" % chan_num,
                "--target-rangeidx %d" % vv.target_rangeidx,
                "--rblock-size %d" % vv.rblock_size]
    all_commands.append(" ".join(gen_command + gen_args))

all_commands.append("wait")
#%% Useful function to compute blocks
def getBlocks(N, b):
    return [(x, x + b if x < N-b else N) for x in range(0,N,b)]
    
#%% Generate commands to multi channel process the data
r_blocks = getBlocks(r_sys.Nr, vv.rblock_size)
gen_command = ["python -m processMultiChannel"]
for blks in r_blocks:
    gen_args = ["--config-xml %s" % vv.config_xml,
                "--ridx %d %d" % blks,
                "--xblock-size %d &" % vv.xblock_size]
    all_commands.append(" ".join(gen_command + gen_args))

all_commands.append("wait")    
#%% Generate commands to SAR (w-k) process the data
x_blocks = getBlocks(len(r_sys.ks_full), vv.xblock_size)
gen_command = ["python -m sarProcess"]
for blks in x_blocks:
    gen_args = ["--config-xml %s" % vv.config_xml,
                "--xidx %d %d" % blks,
                "--rblock-size %d &" % vv.rblock_size]
    all_commands.append(" ".join(gen_command + gen_args))

all_commands.append("wait")

#%% Write the commands to file
cmd_file = ".".join(vv.config_xml.split(".")[0:-1] + ["sh"])
with open(cmd_file, 'w') as f:
    for cmd in all_commands:
        f.write(cmd + "\n")