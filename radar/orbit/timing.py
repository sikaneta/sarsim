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
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm

#%% Define some state vector data
svtime=np.datetime64("2021-10-25T09:27:12.000000")
svdata = np.array([-3955037.876120, 5461346.916367, 2138635.863933,
                   2628.185608, -880.260804, 7074.139453])

#%% Create the state vector object
sv = state_vector()
sv.add(svtime, svdata)

#%% Generate some data
elev = -np.radians(np.arange(0.01,60,0.01))
ranges, rhat, inc, tau, swath = getTiming(sv, elev)


#%% Plot the ranges
fp1 = 1000
x = swath/1e3
k = 5
y = tau + k/fp1
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

fig, axs = plt.subplots()
axs.set_xlim(x.min(), x.max())
axs.set_ylim(y.min(), y.max())
norm = plt.Normalize(y.min(), y.max())
lc = LineCollection(segments, cmap='viridis', norm=norm)

lc.set_array(y)
lc.set_linewidth(2)
line = axs.add_collection(lc)
fig.colorbar(line, ax=axs)

#%%
Np = 10
x = swath/1e3
idxs = [np.argwhere(np.abs(x-770)<k) for k in np.arange(140,20,-10)]
plt.figure()
for k in range(Np):
    y = tau + k/fp1
    plt.plot(x, y*1e3, 'lightgrey')
    for kk, idx in enumerate(idxs):
        plt.plot(x[idx], 
                 y[idx]*1e3, 
                 c=cm.hot(kk/len(idxs)),
                 linewidth=2.5)
    plt.axhline(y=y[0]*1e3, color='plum', linewidth = 3)
#plt.grid()
plt.xlabel('Ground range (km)')
plt.ylabel('Fast-time (ms)')
plt.ylim(0.004*1e3,0.0150*1e3)
plt.show()

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
plt.ylabel('Fast-time (ms)')
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