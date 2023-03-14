#%% -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:10:22 2022

@author: Ishuwa.Sikaneta
"""

from space.planets import venus
from orbit.pointing import simulation
from orbit.envision import loadSV

import numpy as np
import matplotlib.pyplot as plt
import json
import os


#%% Create a simulation object
eSim = simulation(planet = venus(),
                  e_ang = 14.28,
                  azAxis = 5.5,
                  elAxis = 0.6125,
                  carrier = 3.15e9)

#%% Load an oem orbit state vector file from Venus
sv = loadSV()

#%% Define sample simulation parameters
""" Select a subset of state vectors """
selection_range = [270, 480, 2]
mysvs = eSim.state(sv.measurementData, selection_range)
 
times = [(sv.measurementTime[k] - sv.measurementTime[0])/np.timedelta64(1, 's') 
         for k in range(*selection_range)]

xidx = 0
X = mysvs[1]

#%% Define the geometry
""" Define the off-nadir angle. Negative angles for right looking """
off_nadir = 18.7

""" Generate the covariances """  
covariances = {
    "spacecraft": {
        "description": "Errors in the orientation of the spacecraft.",
        "referenceVariables": "RollPitchYaw",
        "units": "radians",
        "R": ((0*np.diag([8.2e-3, 0.93e-3, 0.93e-3])/2)**2).tolist()
        },
    "instrument": {
        "description": """Errors in the pointing of the antenna. From JPL
                          spreadsheet cells 'SAR APE Pointing Budget'!D34:36
                          These are Allocation values and do not include a 20% 
                          margin.""",
        "referenceVariables": "AzimuthElevationTilt",
        "units": "radians",
        "R": ((0*np.diag([0.65e-3, 4.50e-3, 0.52e-3])/2)**2).tolist()
        },
    "orbitVelocity": {
        "description": "Errors in the orbit velocity vector",
        "referenceVariables": "VxVyVz",
        "units": "m/s",
        "R": (0*np.diag([0.2, 0.2, 0.2])**2).tolist()
        },
    "orbitAlongTrack": {
        "description": "Error in the orbit time (error in orbit angle)",
        "referenceVariables": "t",
        "units": "s",
        "R": (4.3/2)**2
        },
    "orbitAcrossTrack": {
        "description": "Error in the orbit across track position (orbit tube)",
        "referenceVariables": "dX",
        "units": "m",
        "R": np.diag([(600/2)**2, (850/2)**2])
        }
    }
    
#%% Run the simulation on the defined values
R, AEU_m = eSim.contributors2aeuCovariance(X, off_nadir, covariances)

#%%
R_AEU = AEU_m.dot(R).dot(AEU_m.T)

#%%
R_RPY = eSim.aeu2rpyCovariance(R_AEU)

#%%
print(2*np.sqrt(np.diag(R_RPY)))