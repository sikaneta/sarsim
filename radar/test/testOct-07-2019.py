# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:59:43 2019

@author: SIKANETAI
"""

import sys
import getopt
import numpy as np
from datetime import datetime, date, time, timedelta
from math import pi
import urllib
from copy import deepcopy
import lxml.etree as etree
from functools import reduce
import os
import argparse
import generateXML.sure as s_config
import configuration.configuration as cfg
import multichannel.msar as msar
import matplotlib.pyplot as plt
import omegak.omegak as wk
import scipy.signal.windows as s_win

#%%
resolution = 0.3
res_str = "%dcm" % int(resolution*100) 

#%%
if "HOME" in os.environ.keys():
    home = os.environ["HOME"]
elif "HOMEPATH" in os.environ.keys():
    home = os.environ["HOMEPATH"]
else:
    home = ""
print("Home path: %s" % home)
home = "E:%s" % home

#%%
default_simulation_file = os.path.sep.join([home,
                                            "simulation",
                                            res_str,
                                            "%s_simulation.xml" % res_str])
class fake_parse:
    def __init__(self,
                 xml_file = default_simulation_file,
                 az_resolution = 1.0,
                 rn_resolution = 1.0,
                 swath_width = 10000.0,
                 rn_oversample = 1.5,
                 az_oversample = 1.5,
                 pulse_duration = 20.5,
                 max_antenna_length = 25.0,
                 compressed_range_samples = 1024,
                 file_data_domain = "rx"):
        self.xml_file = xml_file
        self.az_resolution = az_resolution
        self.rn_resolution = rn_resolution
        self.swath_width = swath_width
        self.rn_oversample = rn_oversample
        self.az_oversample = az_oversample
        self.pulse_duration = pulse_duration
        self.max_antenna_length = max_antenna_length
        self.file_data_domain = file_data_domain
        self.compressed_range_samples = compressed_range_samples
        
#%%
vv = fake_parse(az_resolution = resolution,
                rn_resolution = resolution,
                compressed_range_samples = 2048)

#%%
newxml = os.path.sep.join([home,
                           "simulation",
                           res_str,
                           "reduced_%s_simulation.xml" % res_str])
#%% Load the configuration
radar = cfg.loadConfiguration(newxml)

#%% Define which bands to look at
bands = np.arange(-int(len(radar)/2)-2,int(len(radar)/2)+1+2)

#%% Compute the ground point and associate slow time parameters
r_sys = cfg.radar_system(radar, bands)
r_sys.computeGroundPoint(radar, range_idx=400)
print(r_sys.kr[0])
print(r_sys.ks_full[0])
print(r_sys.C.a2)

#%%
s = np.arange(r_sys.Na*r_sys.n_bands)/(r_sys.ksp*r_sys.n_bands)
s = s - np.mean(s)
cdf = r_sys.C.cdf

# Compute the range curve
s_off = 0.0*r_sys.C.ds((r_sys.target_time - r_sys.expansion_time)/np.timedelta64(1,'s'))
rngs_curve = (np.outer(cdf[0] - r_sys.target_ground_position, (s-s_off)**0) 
              + np.outer(cdf[1], (s-s_off)) 
              + np.outer(cdf[2], ((s-s_off)**2)/2.0) 
              + np.outer(cdf[3], ((s-s_off)**3)/6.0))
rngs = np.sqrt(np.sum(rngs_curve*rngs_curve, axis=0))

C = r_sys.C

sx = r_sys.C.ds((r_sys.target_time - r_sys.expansion_time)/np.timedelta64(1,'s'))
s0 = 0.0

r = np.linalg.norm(C.R)
a = s - s0
b = (sx - s0)

aT = 1 - C.kappa**2*(a**2 + a*b + b**2)/6
aN = C.kappa*(a + b)/2 + C.dkappa*(a**2 + a*b + b**2)/6
aB = C.kappa*C.tau*(a**2 + a*b + b**2)/6

rcos = np.dot(-C.R, C.N)
rsin = np.dot(-C.R, C.B)

cT = (a-b)*aT + b*C.kappa*rcos
cN = (a-b)*aN - rcos + b*C.tau*rsin
cB = (a-b)*aB - rsin - b*C.tau*rcos

vecs = np.outer(C.T, cT) + np.outer(C.N, cN) + np.outer(C.B, cB)

nrng = np.sqrt(np.sum(vecs*vecs, axis=0))

a0 = b**2*C.kappa**2*rcos**2 + b**2*C.tau**2*r**2 + r**2
a1 = 2*b**2*C.tau/3*(b*C.dkappa+2*C.kappa)*rsin - 2*b**2/3*(b*C.kappa*C.tau**2 + b*C.kappa**3 + C.dkappa)*rcos
a2 = (1 - (C.kappa+b*C.dkappa + b**2*(C.kappa**3+C.kappa*C.tau**2))*rcos 
      + b**2*C.dkappa*C.tau*rsin + b**4/9*(C.kappa**4+C.kappa**2*C.tau**2+C.dkappa**2) 
      + b**2/3*C.kappa*(C.kappa + 2*b*C.dkappa))
a3 = (-1/3*(C.kappa*C.tau*rsin + C.dkappa*rcos) 
     + b**3/3*(C.kappa**4 + C.kappa**2*C.tau**2 + C.dkappa**2)
     + 4*b**2/3*(C.kappa*C.dkappa)
     + b/3*(C.dkappa*C.tau*rsin - (C.kappa*C.tau**2 + C.kappa**3)*rcos))
a4 = -C.kappa**2/12 + 13*b**2/36*(C.kappa**4 + C.kappa**2*C.tau**2 + C.dkappa**2) + 5*b*C.kappa*C.dkappa/6
a5 = b/6*(C.kappa**4 + C.kappa**2*C.tau**2 + C.dkappa**2 + C.kappa*C.dkappa)
a6 = 1/36*(C.kappa**4 + C.kappa**2*C.tau**2 + C.dkappa**2)

prng = np.sqrt(a0 
               + a1*(s-sx) 
               + a2*(s-sx)**2 
               + a3*(s-sx)**3 
               + a4*(s-sx)**4 
               + a5*(s-sx)**5 
               + a6*(s-sx)**6)

crng = np.sqrt(r**2 + C.a2*(s-sx)**2 + C.a3*(s-sx)**3 + C.a4*(s-sx)**4)

vcts = s*(C.T + s/2*C.kappa*C.N + s**2/6*(-C.kappa**2*C.T + C.dkappa*C.N + C.kappa*C.tau*C.B))

#%%
head, tail = os.path.split(radar[0]['filename'])
output_file = "_".join(tail.split("_")[0:1] + ["xr", "mchanprocessed.npy"])
output_file = os.path.sep.join([os.path.split(head)[0],
                                "l1_data",
                                output_file])
    
#%%
#print("Loading: %s" % output_file)
#procData = cfg.loadNumpy_mcp_data(output_file)






