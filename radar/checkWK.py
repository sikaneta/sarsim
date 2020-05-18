import numpy as np
import os
import argparse
import configuration.configuration as cfg
import utils.fileio as fio
from glob import glob

#%% Parse the arguments
parser = argparse.ArgumentParser(description="""Check for properly
                                                processed files""")
parser.add_argument("--config-xml",
                    help="The config XML file to test against",
                    required=True)
parser.add_argument("--xblock-size",
                    help="Size of the output data block in the X direction",
                    type=int,
                    required=True)
parser.add_argument("--rblock-size",
                    help="Size of data blocks in the r direction",
                    type=int,
                    required=True)
parser.add_argument("--number-processors",
                    help="The number of processors",
                    type=int,
                    default=4)
parser.add_argument("--vanilla-stolt",
                    help="Apply vanilla Stolt interpolation a3=a4=0",
                    action="store_true")
parser.add_argument("--output-scriptfile",
                    help="The name of the output script file",
                    default="./missing_wk.sh")

#vv = parser.parse_args("--config-xml 12cm_sim.xml --vanilla-stolt --rblock-size 256 --xblock-size 8192".split())
vv = parser.parse_args()

#%% Make sure we're looking at the absolute path
vv.config_xml = os.path.abspath(vv.config_xml)

#%% Load the radar object
radar = cfg.loadConfiguration(vv.config_xml)

#%% Load the r_sys object
r_sys = cfg.loadRsys(radar)

#%% Compute expected number of samples in azimuth
Na = len(r_sys.ks_full)
ref_rng = range(0,Na,vv.xblock_size)

#%% Compute the name signature of the processed data
processed_file = fio.fileStruct(radar[0]['filename'],
                                "wk_processed",
                                "X*r0",
                                "wkprocessed.npy")
flist = glob(processed_file)

#%% Get the signature from the list
fsigs = [os.path.split(x)[-1].split("_")[2] for x in flist]
fazi = sorted([int(x[1:-2]) for x in fsigs])

#%% Find which signatures are missing
missing = [x for x in ref_rng if x not in fazi]

#%% Generate new commands with the missing data signature
x_arg = ["--xidx %d %d" % (x, x+vv.xblock_size) for x in missing]

#%% SBATCH commands
vsc_comm = ["#SBATCH -J sarsim", "#SBATCH -N 1", "eval \"$(conda shell.bash hook)\"", "conda activate radar"]
sarproc_commands = ["#!/bin/bash"] + vsc_comm

#%% Loop through missing files
#%% Generate commands to SAR (w-k) process the data
gen_command = ["python -m sarProcess"]
for blk, k in zip(x_arg, range(len(x_arg))):
    if k%vv.number_processors == 0:
        sarproc_commands.append("wait")
    gen_args = ["--config-xml %s" % vv.config_xml,
                blk,
                "--rblock-size %d &" % vv.rblock_size]
    if vv.vanilla_stolt:
        gen_args = ["--vanilla-stolt"] + gen_args
    sarproc_commands.append(" ".join(gen_command + gen_args))

sarproc_commands.append("wait")

#%% Write to file
with open(vv.output_scriptfile, 'w') as f:
    for cmd in sarproc_commands:
        f.write(cmd + "\n")

