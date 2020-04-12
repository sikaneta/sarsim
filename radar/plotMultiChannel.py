#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:38:54 2020

@author: ishuwa
"""

import configuration.configuration as cfg
import plot.simplot as sp
import numpy as np
import argparse
import os
import utils.fileio as fio
from tqdm import tqdm
import matplotlib.pyplot as plt

#%% Argparse stuff
purpose = """
          Plot some figures from the multichannel processed
          data. These can be used to check the quality of the
          processing
          """
parser = argparse.ArgumentParser(description=purpose)

parser.add_argument("--config-xml", help="The config XML file", 
                    default = u'/home/ishuwa/simulation/20cm/simulation_20cm.xml')
parser.add_argument("--mchan-processed-file",
                    help="The name of the multi-channel processed output file",
                    default = None)
parser.add_argument("--target-range-index",
                    help="""The range sample index of the target.
                            This is needed to compute the reference phase
                            for the last phase plot""",
                    type=int,
                    default=None)
parser.add_argument("--ridx",
                    help="The range indeces to examine",
                    type=int,
                    nargs=2,
                    default=[0,None])
parser.add_argument("--xblock-size",
                    help="Size of data blocks in the X direction",
                    type=int,
                    default=16384)
parser.add_argument("--arclength-offset",
                    help="Offset of the target from zero in arclength (m)",
                    type=float,
                    default=0.0)
parser.add_argument("--interactive",
                    help="Interactive mode. Program halts until figures are closed",
                    action="store_true",
                    default=False)
            
vv = parser.parse_args()

#%% Load the radar configuration
if 'radar' not in locals():
    radar = cfg.loadConfiguration(vv.config_xml)

#%% Compute the ground point and associate slow time parameters
r_sys = cfg.loadRsys(radar)

#%% Useful function to compute blocks
def getBlocks(N, b):
    return [(x, x + b if x < N-b else N) for x in range(0,N,b)]


#%% Load the data
if vv.mchan_processed_file is None:
    vv.mchan_processed_file = fio.fileStruct(radar[0]['filename'],
                                             "mchan_processed",
                                             "Xr",
                                             "mchanprocessed.npy")

#%% Define the data blocks for loading
Na = len(r_sys.ks_full)
x_blocks = getBlocks(Na, vv.xblock_size)

#%% Load the data summed across range
signal = np.zeros((Na,), dtype=np.complex128)
for k in tqdm(range(len(x_blocks))):
    xidx = x_blocks[k]
    signal[xidx[0]:xidx[1]] = np.sum(fio.loadSimFiles(vv.mchan_processed_file, 
                                                      xidx=xidx, 
                                                      ridx=vv.ridx), axis=1)

#%%
arc_signal = np.fft.ifft(signal)

#%% Compute the ground point and associate slow time parameters
if vv.target_range_indx is not None:
    r_sys.computeGroundPoint(radar, range_idx=vv.target_range_index)

#%%
def computeReference(r_sys, arc_signal):
    # Compute the slow time parameter
    s = np.arange(r_sys.Na*r_sys.n_bands)/(r_sys.ksp*r_sys.n_bands)
    s = s - np.mean(s)
    cdf = r_sys.C.cdf
    r = np.linalg.norm(r_sys.C.R)
    
    # Compute the range curve
    rngs_curve = np.outer(r_sys.C.R, s**0) + np.outer(cdf[1],s) + np.outer(cdf[2], (s**2)/2.0) + np.outer(cdf[3], (s**3)/6.0)
    rngs = np.sqrt(np.sum(rngs_curve*rngs_curve, axis=0))
    rngs = np.sqrt(r**2 + r_sys.C.a2*s**2 + r_sys.C.a3*s**3 + r_sys.C.a4*s**4)
    
    # Create the inverse phase function
    rC = np.exp(-1j*r_sys.kr[0]*rngs)
    
    # Compute the offset in s
    dummy = arc_signal*np.conj(rC)
    phs_diff = np.angle(np.sum(dummy[1:]*np.conj(dummy[0:-1])))
    ds = r*phs_diff/r_sys.kr[0]/r_sys.C.a2/(s[1]-s[0])
    
    # Recalculate with new s
    s -= ds
    
    # Compute the range curve
    rngs_curve = np.outer(r_sys.C.R, s**0) + np.outer(cdf[1],s) + np.outer(cdf[2], (s**2)/2.0) + np.outer(cdf[3], (s**3)/6.0)
    rngs_full = np.sqrt(np.sum(rngs_curve*rngs_curve, axis=0))
    rngs_approx = np.sqrt(r**2 + r_sys.C.a2*s**2 + r_sys.C.a3*s**3 + r_sys.C.a4*s**4)
    
    rngs = rngs_approx
    
    # Create the inverse phase function
    rC = np.exp(-1j*r_sys.kr[0]*rngs)
    
    # Cacluate the approximation
    rsphi = r_sys.C.R.dot(r_sys.C.B)
    rcphi = r_sys.C.R.dot(r_sys.C.N)
    a1 = 2*(r_sys.sx**2)/3*(2*r_sys.C.tau*r_sys.C.kappa*rsphi 
                            +r_sys.C.tau*r_sys.C.dkappa*r_sys.sx*rsphi -
                            r_sys.C.dkappa*rcphi -
                            r_sys.C.kappa*r_sys.C.tau**2*r_sys.sx*rcphi -
                            r_sys.C.kappa**3*r_sys.sx*rcphi)
    a6 = 1/36*(r_sys.C.kappa**4 + 
               r_sys.C.kappa**2*r_sys.C.tau**2 +
               r_sys.C.dkappa**2)
    sx = s - 0.39198 #0*r_sys.sx
    rngs_approx = np.sqrt(r**2 + 
                          a1*sx +
                          r_sys.C.a2*sx**2 + 
                          r_sys.C.a3*sx**3 + 
                          r_sys.C.a4*sx**4 +
                          a6*sx**6)
    
    return s, rC, rngs_full, rngs_approx 

s, rC, rngs_left, rngs_right = computeReference(r_sys, arc_signal)
plt.figure()
plt.plot(s, rngs_left - rngs_right)
plt.grid()
plt.show()   

#%%
proc_signal = arc_signal*np.conj(rC)
ang = np.unwrap(np.angle(proc_signal))
half_n_ang = int(len(ang)/2)
D = 0.2

plt.figure()
plt.plot(s, ang)
plt.grid()
plt.ylim([ang[half_n_ang]-D, ang[half_n_ang]+D])
plt.show()

plt.figure()
plt.plot(s, (np.abs(proc_signal)))
plt.grid()
plt.show()

#%% Define the folder in which to store plots
sim_folder = os.path.join(os.path.split(vv.config_xml)[0], 
                          "simulation_plots")

#%% Do the plotting
# sp.mchanPlot(procData, 
#              r_sys, 
#              vv.interactive, 
#              sim_folder, 
#              vv.arclength_offset)