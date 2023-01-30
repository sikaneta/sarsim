# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 14:22:02 2022

@author: ishuwa.sikaneta
"""

from orbit.geometry import getTiming
from orbit.envision import loadSV

import numpy as np
import matplotlib.pyplot as plt

#%% Load state vectors from envision
sv = loadSV()

#%% Generate some data
off_nadir = np.arange(0.01,60,0.1)
incidence = np.zeros((len(off_nadir), 100), dtype = float)

#%%
for k in range(100):
    _, _, incidence[:,k], _, _ = getTiming(sv, np.radians(off_nadir), idx = k)

#%% Plot the data
plt.figure()
plt.plot(off_nadir, incidence)
plt.xlabel("Off-nadir angle (deg)")
plt.ylabel("Incidence angle (deg)")
plt.grid()
plt.show()