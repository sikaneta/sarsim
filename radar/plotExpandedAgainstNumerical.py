#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:38:54 2020

@author: ishuwa
"""

import configuration.configuration as cfg
import plot.simplot as sp
import numpy as np
from scipy.signal import hann as weightwin
import argparse
import os
import utils.fileio as fio
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.constants import Boltzmann, c
from scipy import interpolate

#%% Argparse stuff
purpose = """
          Plot some figures from the SAR processed
          data. These can be used to check the quality of the
          processing
          """
parser = argparse.ArgumentParser(description=purpose)

parser.add_argument("--config-xml", 
                    help="The config XML file", 
                    required=True)

parser.add_argument("--arclength-offset",
                    help="Offset of the target from zero in arclength (m)",
                    type=float,
                    default=0.0)
parser.add_argument("--ridx",
                    help="The range indeces to examine",
                    type=int,
                    nargs=2,
                    default=[0,None])
parser.add_argument("--xidx",
                    help="The azimuth indeces to examine",
                    type=int,
                    nargs=2,
                    default=[0,None])
parser.add_argument("--interactive",
                    help="Interactive mode. Program halts until figures are closed",
                    action="store_true",
                    default=False)

#%% Parse the arguments           
vv = parser.parse_args('--config-xml /home/ishuwa/simulation/12cm/12cm.xml'.split())

#%% Load the radar configuration
if 'radar' not in locals():
    radar = cfg.loadConfiguration(vv.config_xml)
    
#%% Get an r_sys object
r_sys = cfg.loadRsys(radar)

#%% Get the ground position
X = r_sys.target_ground_position

#%% Get the slow time object and arc length
C = radar[0]['acquisition']['satelliteArc'][1]
s = np.array(radar[0]['acquisition']['satelliteArc'][0])

# #%% Compute the wavelength of carrier
wavelength = c/r_sys.f0

# Get the satpositions from integration and from expansion
sPos_numerical = radar[0]['acquisition']['satellitePositions'][1][:,0:3].T
sVel_numerical = radar[0]['acquisition']['satellitePositions'][1][:,3:6].T
sPos_expanded = (np.outer(C.cdf[0], np.ones_like(s)) +
                 np.outer(C.cdf[1], s) +
                 np.outer(C.cdf[2], s**2)/2.0 +
                 np.outer(C.cdf[3], s**3)/6.0)
sDer_expanded = (np.outer(C.cdf[1], np.ones_like(s)) +
                 np.outer(C.cdf[2], s) +
                 np.outer(C.cdf[3], s**2)/2.0)
sPos_delta = sPos_numerical - sPos_expanded

# Calculate the domain variable
Rn_vector = sPos_numerical - np.outer(X, np.ones_like(s))
Rn = np.linalg.norm(Rn_vector, axis=0)
rhat_vector = Rn_vector/np.outer(np.ones((3,)), Rn)
ksM = -r_sys.kr[0]*np.sum(rhat_vector*sDer_expanded, axis=0)

# Caluclate component of delta in direction of rhat (dependent variable)
delta = -r_sys.kr[0]*np.sum(sPos_delta*rhat_vector, axis=0)

# Compute the interpolator
f = interpolate.interp1d(ksM, delta, kind = 'linear')

# Get the ks dependent values
phase = f(r_sys.ks_full)

#%% Plot the difference as a function of delta
plt.figure()
plt.plot(r_sys.ks_full, phase)
plt.grid()
plt.show()

#%% Compute the range as calculated from state_vectors
Rn = np.linalg.norm(radar[0]['acquisition']['satellitePositions'][1][:,0:3] -
                    np.outer(np.ones_like(s), X), axis=1)

#%% Get the arc length computed satellite positions

Rs = np.linalg.norm(np.outer(C.cdf[0] - X, np.ones_like(s)) +
                    np.outer(C.cdf[1], s) +
                    np.outer(C.cdf[2], s**2)/2.0 +
                    np.outer(C.cdf[3], s**3)/6.0, axis=0)

plt.figure()
plt.title("Difference between numerical and expanded range")
plt.plot(s,(Rn - Rs)/wavelength*2*np.pi)
plt.grid()
plt.xlabel('arclength (m)')
plt.ylabel('Phase difference (rad)')
plt.show()

#%% Second expansion
a0 = np.linalg.norm(r_sys.C.R)**2
sx = r_sys.sx - 0.558
Ra = np.sqrt(a0 + 
             r_sys.C.a2*(s - sx)**2 +
             r_sys.C.a3*(s - sx)**3 +
             r_sys.C.a4*(s - sx)**4)

plt.figure()
plt.title("Difference between expansion and approximation")
plt.plot(s,Rs - Ra)
plt.grid()
plt.xlabel('arclength (m)')
plt.ylabel('Expanded - Approximated (m)')
plt.show()

