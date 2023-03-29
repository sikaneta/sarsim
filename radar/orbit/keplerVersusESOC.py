# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:18:27 2022

@author: ishuwa.sikaneta
"""

from space.planets import venus
from orbit.orientation import orbit
import numpy as np
import matplotlib.pyplot as plt
from orbit.envision import loadSV

#%% Load state vectors for envision
sv = loadSV(toPCR = False)

#%% Run a test#%% Define a reference state vector to use
""" First get the state vector """
svsidx = np.arange(8270, 8400, 1)
X = sv.measurementData[svsidx[0]]
tref = sv.measurementTime[svsidx[0]]

#%% Define the vensar object and compute the orbit angle
venSAR = orbit(planet=venus(), angleUnits="degrees")
orbitAngle, ascendingNode = venSAR.setFromStateVector(X)
t0 = venSAR.computeT(orbitAngle)

#%% Compute the differences to state vector
""" Calculate the difference between the Kepler
    propagated orbit and the state vector from
    ESOC """
odiff = np.array([venSAR.computeSV(venSAR.computeO(t0 + (sv.measurementTime[k] - tref)/np.timedelta64(1,'s'))[0])[0] - sv.measurementData[k] 
                  for k in svsidx])

#%% Plot the results
plt.figure()
plt.plot(odiff)
plt.show()
plt.grid()