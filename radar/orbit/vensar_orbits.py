# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 16:39:38 2022

@author: Ishuwa.Sikaneta
"""


from space.planets import venus
from orbit.orientation import orbit
from orbit.euler import YRPfromRotation
from orbit.euler import aeuAnglesDAEUfromAAEU, aeuAnglesAAEUfromDAEU
import numpy as np
import matplotlib.pyplot as plt
from measurement.measurement import state_vector
from scipy.constants import c
from numpy.random import default_rng
import json
from datetime import datetime as dt

rng = default_rng()
import os


#%% Define some parameters for venSAR
elAxis = 0.6
azAxis = 6.0

#%% Get the state vectors
def getStateVectors(svfile):
    with open(svfile, 'r') as f:
        svecs = f.read().split('\n')
    svecs = [s.split() for s in svecs[16:-1]]
    svecs = [s for s in svecs if len(s)==7]
    svs = [(np.datetime64(s[0]), np.array([float(x) for x in s[1:]])*1e3) 
           for s in svecs]
    
    t = [(s[0] - svs[0][0])/np.timedelta64(1,'s') for s in svs]
    
    sv = state_vector(planet=venus(), harmonicsCoeff=180)
    for k in range(len(t)):
        sv.add(svs[k][0], sv.toPCR(svs[k][1], t[k]))
    
    return sv

#%% Load an oem orbit state vecctor file from Venus
orb_path = r"C:\Users\Ishuwa.Sikaneta\local\Data\Envision"
svfile = os.path.join(orb_path, "EnVision_ALT_T4_2032_SouthVOI.oem")
sv = getStateVectors(svfile)

#%%
xidx = 200
S = sv.measurementData[xidx]
u = 0.0
v = -np.cos(np.radians(30))
X = sv.computeGroundPosition(S, u, v)
Sx = S[:3]
Sv = S[3:]
R = X - Sx

#%% Set the Kepler parameters
venSAR = orbit(planet=venus(), angleUnits="radians")
orbitAmgle, ascendingNode = venSAR.setFromStateVector(sv.measurementData[xidx])

#%% Compute the broadside position one orbit later
t0 = sv.measurementTime[xidx]
t1 = t0 + np.timedelta64(int(venSAR.period*1e9), 'ns')
bTime, bSv, bError = sv.computeBroadsideToX(t1, X)

#%% Calculate the differences
dSx = sv.measurementData[xidx][:3] - bSv[:3]
dSv = sv.measurementData[xidx][3:] - bSv[3:]