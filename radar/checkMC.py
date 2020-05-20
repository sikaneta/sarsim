import numpy as np
import os
import argparse
import configuration.configuration as cfg
import utils.fileio as fio
from glob import glob

#%% Parse the arguments
parser = argparse.ArgumentParser(description="""Check for properly
                                                processed mChan files""")
parser.add_argument("--config-xml",
                    help="The config XML file to test against",
                    required=True)
parser.add_argument("--xblock-size",
                    help="Size of the output data block in the X direction",
                    type=int,
                    required=True)
parser.add_argument("--xblock",
                    help="X block number",
                    type=int,
                    default=0)
parser.add_argument("--rblock-size",
                    help="Size of data blocks in the r direction",
                    type=int,
                    required=True)
parser.add_argument("--number-processors",
                    help="The number of processors",
                    type=int,
                    default=4)
parser.add_argument("--output-scriptfile",
                    help="The name of the output script file",
                    default="./missing_mchan.sh")
parser.add_argument("--folders",
                    help="Folders with processed data",
                    nargs="*",
                    default=[])

#vv = parser.parse_args("--config-xml 12cm_sim.xml --vanilla-stolt --rblock-size 256 --xblock-size 8192".split())
vv = parser.parse_args()

#%% Make sure we're looking at the absolute path
vv.config_xml = os.path.abspath(vv.config_xml)

#%% Load the radar object
radar = cfg.loadConfiguration(vv.config_xml)

#%% Load the r_sys object
r_sys = cfg.loadRsys(radar)

#%% Compute expected number of samples in azimuth
Nr = len(r_sys.kr)
ref_rng = range(0,Nr,vv.rblock_size)
ref_azi = range(0, len(r_sys.ks_full), vv.xblock_size)

missing = []
for xb in ref_azi:
    #%% Compute the name signature of the processed data
    processed_file = fio.fileStruct(radar[0]['filename'],
                                    "mchan_processed",
                                    "X%dr*" % xb,
                                    "mchanprocessed.npy")
    flist = glob(processed_file)
    
    for fld in vv.folders:
        flist += glob(os.path.join(fld, "10cm_sim_X%dr*_mchanprocessed.npy" % xb))
    
    #%% Get the signature from the list
    fsigs = [os.path.split(x)[-1].split("_")[2] for x in flist]
    frng = sorted([int(x.split('r')[-1]) for x in fsigs])
    
    #%% Find which signatures are missing
    miss = [x for x in ref_rng if x not in frng]
    
    if len(miss)==0:
        print("[*] No files missing for xblock: %d" % xb)
    else:
        print("[ ] Files missing for xblock: %d" % xb)
        missing += miss

#%% Keep only unique values
missing = list(set(missing))

#%% Generate new commands with the missing data signature
x_arg = ["--ridx %d %d" % (x, x+vv.rblock_size) for x in missing]

#%% SBATCH commands
vsc_comm = ["#SBATCH -J sarsim", "#SBATCH -N 1", "eval \"$(conda shell.bash hook)\"", "conda activate radar"]
sarproc_commands = ["#!/bin/bash"] + vsc_comm

#%% Loop through missing files
#%% Generate commands to SAR (w-k) process the data
gen_command = ["python -m processMultiChannel"]
for blk, k in zip(x_arg, range(len(x_arg))):
    if k%vv.number_processors == 0:
        sarproc_commands.append("wait")
    gen_args = ["--config-xml %s" % vv.config_xml,
                blk,
                "--xblock-size %d &" % vv.xblock_size]
    sarproc_commands.append(" ".join(gen_command + gen_args))

sarproc_commands.append("wait")

#%% Write to file
with open(vv.output_scriptfile, 'w') as f:
    for cmd in sarproc_commands:
        f.write(cmd + "\n")

