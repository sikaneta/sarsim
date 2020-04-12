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
parser.add_argument("--ref-rangeidx",
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
parser.add_argument("--number-processors",
                    help="The number of processors",
                    type=int,
                    default=4)
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
    r_sys = cfg.computeStoreRsys(radar, bands, vv.ref_rangeidx)
else:
    r_sys = cfg.loadRsys(radar) or cfg.computeStoreRsys(radar, 
                                                        bands, 
                                                        vv.ref_rangeidx)

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
gensignal_commands = ["#!/bin/bash"]
mproc_commands = ["#!/bin/bash"]
sarproc_commands = ["#!/bin/bash"]

#%% Generate commands to create the raw signal
gen_command = ["python -m generateMsar"]
for chan_num in range(len(radar)):
    gen_args = ["--config-xml %s" % vv.config_xml,
                "--channel-idxs %d" % chan_num,
                "--target-rangeidx %d" % vv.target_rangeidx,
                "--rblock-size %d" % vv.rblock_size]
    all_commands.append(" ".join(gen_command + gen_args))
    gensignal_commands.append(" ".join(gen_command + gen_args))

#%% Useful function to compute blocks
def getBlocks(N, b):
    return [(x, x + b if x < N-b else N) for x in range(0,N,b)]
    
#%% Generate commands to multi channel process the data
r_blocks = getBlocks(r_sys.Nr, vv.rblock_size)
gen_command = ["python -m processMultiChannel"]
for blks, k in zip(r_blocks, range(len(r_blocks))):
    if k%vv.number_processors == 0:
        all_commands.append("wait")
        mproc_commands.append("wait")
    gen_args = ["--config-xml %s" % vv.config_xml,
                "--ridx %d %d" % blks,
                "--xblock-size %d &" % vv.xblock_size]
    all_commands.append(" ".join(gen_command + gen_args))
    mproc_commands.append(" ".join(gen_command + gen_args))

all_commands.append("wait")
mproc_commands.append("wait")
 
#%% Generate commands to SAR (w-k) process the data
x_blocks = getBlocks(len(r_sys.ks_full), vv.xblock_size)
gen_command = ["python -m sarProcess"]
for blks, k in zip(x_blocks, range(len(x_blocks))):
    if k%vv.number_processors == 0:
        all_commands.append("wait")
        sarproc_commands.append("wait")
    gen_args = ["--config-xml %s" % vv.config_xml,
                "--xidx %d %d" % blks,
                "--rblock-size %d &" % vv.rblock_size]
    all_commands.append(" ".join(gen_command + gen_args))
    sarproc_commands.append(" ".join(gen_command + gen_args))

all_commands.append("wait")
sarproc_commands.append("wait")

#%% Write the commands to file
bash_prefixes = [("run_simulation_", all_commands), 
                 ("signal_compute_", gensignal_commands), 
                 ("multichannel_compute_", mproc_commands),
                 ("omegak_compute_", sarproc_commands)]
cmd_file = ".".join(vv.config_xml.split(".")[0:-1] + ["sh"])
head, tail = os.path.split(cmd_file)
for bp in bash_prefixes:
    with open(os.path.join(head, bp[0] + tail), 'w') as f:
        for cmd in bp[1]:
            f.write(cmd + "\n")