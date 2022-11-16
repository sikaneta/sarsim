# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:02:00 2022

@author: ishuwa.sikaneta
"""

from measurement.measurement import state_vector
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

#%% Define the range of elevation angles to look at
def getTiming(sv, elev, idx = 0):
    svdata = sv.measurementData[idx] 
    N = -svdata[:3]/np.linalg.norm(svdata[:3])
    T = svdata[3:]/np.linalg.norm(svdata[3:])
    C = np.cross(N, T)
    C = C/np.linalg.norm(C)
    T = np.cross(C,N)
    T = T/np.linalg.norm(T)
    uhats = np.array([np.cos(eang)*N + np.sin(eang)*C for eang in elev])

    # Calculate range vectors
    rangeVectors = sv.computeRangeVectorsU(svdata, uhats)

    # Calculate the ranges
    ranges = np.linalg.norm(rangeVectors, axis=1)
    rhat = rangeVectors*np.tile(1/ranges, (3,1)).T
    
    # Calculate the times
    tau = 2*ranges/c

    # Calculate the incidence angles
    Xg = np.tile(svdata[:3], (len(elev),1)) + rangeVectors
    XgSwath = np.insert(np.linalg.norm(Xg[1:,:] - Xg[:-1,:], axis=1).cumsum(), 0, 0,0)
    Xg = Xg*np.tile(1/np.linalg.norm(Xg, axis=1), (3,1)).T

    inc = np.degrees(np.arccos(-np.sum(Xg*rhat, axis=1)))
    
    return ranges, rhat, inc, tau, XgSwath

# #%% Generate some data
# elev = np.radians(np.arange(0.01,60,0.001))
# ranges, rhat, inc, tau, swath = getTiming(sv, elev)

# #%% Plot the ranges
# fp1 = 1000
# fp2 = 1000
# Tp = 40e-6
# dTp = 0*200e-6
# Np = 15
# plt.figure()
# for k,l in zip(range(0,Np,2), range(1,Np,2)):
#     plt.plot(swath/1e3, tau + k/fp1, 'r')
#     plt.plot(swath/1e3, tau + k/fp1 + Tp, 'r')
#     plt.axhline(y=tau[0] + k/fp1, color='b')
#     plt.axhline(y=tau[0] + k/fp1 + Tp, color='b')
# plt.grid()
# plt.xlabel('Ground range (km)')
# plt.ylabel('Fast-time (s)')
# plt.show()

# #%% Plot the ranges
# fp1 = 1000
# fp2 = 1000
# Tp = 40e-6
# dTp = 0*200e-6
# Np = 10
# plt.figure()
# for k,l in zip(range(0,Np,2), range(1,Np,2)):
#     plt.plot(swath/1e3, tau + k/fp1, 'r')
#     plt.plot(swath/1e3, tau + k/fp1 + Tp, 'r')
#     plt.axhline(y=tau[0] + k/fp1, color='b')
#     plt.axhline(y=tau[0] + k/fp1 + Tp, color='b')
#     plt.plot(swath/1e3, tau - dTp + l/fp2, 'g')
#     plt.plot(swath/1e3, tau - dTp + l/fp2 + Tp, 'g')
#     plt.axhline(y=tau[0] - dTp + l/fp2, color='b')
#     plt.axhline(y=tau[0] - dTp + l/fp2 + Tp, color='b')
# plt.grid()
# plt.xlabel('Ground range (km)')
# plt.ylabel('Fast-time (s)')
# plt.show()

# #%% Plot the off-nadir angles
# plt.figure()
# for k,l in zip(range(0,Np,2), range(1,Np,2)):
#     plt.plot(elev, tau + k/fp1, 'r')
#     plt.plot(elev, tau + k/fp1 + Tp, 'r')
#     plt.axhline(y=tau[0] + k/fp1, color='b')
#     plt.axhline(y=tau[0] + k/fp1 + Tp, color='b')
#     plt.plot(elev, tau - dTp + l/fp2, 'g')
#     plt.plot(elev, tau - dTp + l/fp2 + Tp, 'g')
#     plt.axhline(y=tau[0] - dTp + l/fp2, color='b')
#     plt.axhline(y=tau[0] - dTp + l/fp2 + Tp, color='b')
# plt.grid()
# plt.xlabel('Off-nadir angle (rad)')
# plt.ylabel('Fast-time (s)')
# plt.show()