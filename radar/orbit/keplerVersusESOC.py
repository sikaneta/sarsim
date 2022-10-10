# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:18:27 2022

@author: ishuwa.sikaneta
"""

from space.planets import venus
from orbit.orientation import orbit
import numpy as np
import matplotlib.pyplot as plt
from measurement.measurement import state_vector
from orbit.envision import envisionState
import os

#%% Load an oem orbit state vecctor file from Venus
orb_path = r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\Envision\Orbits"
svfile = os.path.join(orb_path, "EnVision_ALT_T4_2032_SouthVOI.oem")
with open(svfile, 'r') as f:
    svecs = f.read().split('\n')
svecs = [s.split() for s in svecs[16:-1]]
svecs = [s for s in svecs if len(s)==7]
svs = [(np.datetime64(s[0]), np.array([float(x) for x in s[1:]])*1e3) 
       for s in svecs]

#%% Run a test#%% Define a reference state vector to use
""" First get the state vector """
mysvs = envisionState(svs, [270, 480, 10])
xidx = 0
X = mysvs[xidx][1]

#%% Define the vensar object and compute the orbit angle
venSAR = orbit(planet=venus(), angleUnits="degrees")
orbitAngle, ascendingNode = venSAR.setFromStateVector(X)
t0 = venSAR.computeT(orbitAngle)

#%% Compute the differences to state vector
""" Calculate the difference between the Kepler
    propagated orbit and the state vector from
    ESOC """
odiff = np.array([venSAR.computeR(venSAR.computeO(t0 + k*2*60)[0])[0] - mysvs[k][1] 
                  for k in range(len(mysvs))])

#%% Plot the results
plt.figure()
plt.plot(odiff)
plt.show()
plt.grid()