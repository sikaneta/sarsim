# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:02:00 2022

@author: ishuwa.sikaneta
"""

from measurement.measurement import state_vector
from orbit.geometry import getTiming
import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt

#%% Define some state vector data
svtime=np.datetime64("2021-10-25T09:27:12.000000")
svdata = np.array([-3955037.876120, 5461346.916367, 2138635.863933,
                   2628.185608, -880.260804, 7074.139453])

#%% Create the state vector object
sv = state_vector()
sv.add(svtime, svdata)

#%% Generate some data
elev = np.radians(np.arange(0.01,60,0.001))
ranges, rhat, inc, tau, swath = getTiming(sv, elev)

#%% Plot the ranges
fp1 = 1000
fp2 = 1000
Tp = 40e-6
dTp = 0*200e-6
Np = 15
plt.figure()
for k,l in zip(range(0,Np,2), range(1,Np,2)):
    plt.plot(swath/1e3, tau + k/fp1, 'r')
    plt.plot(swath/1e3, tau + k/fp1 + Tp, 'r')
    plt.axhline(y=tau[0] + k/fp1, color='b')
    plt.axhline(y=tau[0] + k/fp1 + Tp, color='b')
plt.grid()
plt.xlabel('Ground range (km)')
plt.ylabel('Fast-time (s)')
plt.show()

#%% Plot the ranges
fp1 = 1000
fp2 = 1000
Tp = 40e-6
dTp = 0*200e-6
Np = 10
plt.figure()
for k,l in zip(range(0,Np,2), range(1,Np,2)):
    plt.plot(swath/1e3, tau + k/fp1, 'r')
    plt.plot(swath/1e3, tau + k/fp1 + Tp, 'r')
    plt.axhline(y=tau[0] + k/fp1, color='b')
    plt.axhline(y=tau[0] + k/fp1 + Tp, color='b')
    plt.plot(swath/1e3, tau - dTp + l/fp2, 'g')
    plt.plot(swath/1e3, tau - dTp + l/fp2 + Tp, 'g')
    plt.axhline(y=tau[0] - dTp + l/fp2, color='b')
    plt.axhline(y=tau[0] - dTp + l/fp2 + Tp, color='b')
plt.grid()
plt.xlabel('Ground range (km)')
plt.ylabel('Fast-time (s)')
plt.show()

#%% Plot the off-nadir angles
plt.figure()
for k,l in zip(range(0,Np,2), range(1,Np,2)):
    plt.plot(elev, tau + k/fp1, 'r')
    plt.plot(elev, tau + k/fp1 + Tp, 'r')
    plt.axhline(y=tau[0] + k/fp1, color='b')
    plt.axhline(y=tau[0] + k/fp1 + Tp, color='b')
    plt.plot(elev, tau - dTp + l/fp2, 'g')
    plt.plot(elev, tau - dTp + l/fp2 + Tp, 'g')
    plt.axhline(y=tau[0] - dTp + l/fp2, color='b')
    plt.axhline(y=tau[0] - dTp + l/fp2 + Tp, color='b')
plt.grid()
plt.xlabel('Off-nadir angle (rad)')
plt.ylabel('Fast-time (s)')
plt.show()