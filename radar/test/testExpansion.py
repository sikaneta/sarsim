#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:11:00 2020

@author: ishuwa
"""

from measurement.arclength import slow
import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt

#%%
rd = radar[0]
nSamples = rd['acquisition']['numAzimuthSamples']
deltaT = 1.0/rd['acquisition']['prf']

myrd = rd['platform']['stateVectors']
xState = myrd.expandedState(myrd.measurementData[0], 0.0)
reference_time = myrd.measurementTime[0]

ref = rd['acquisition']
np_prf = np.timedelta64(int(np.round(1e9/ref['prf'])),'ns')
svTime = ref['startTime'] + np_prf*ref['numAzimuthSamples']/2.0
print("Reference time:")
print(reference_time)  

# Create a slow time object
C = slow([reference_time])

# Generate the differntial geometry parameters
cdf, tdf, T, N, B, kappa, tau, dkappa = C.diffG(xState)

# Convert time to arclength in the slow-time object
C.t2s()

# Get the arclength corresponding to one PRI
dS = C.ds(deltaT)

#%% Get the arclength relative to the state vector time
start_seconds = (rd['acquisition']['startTime'] - reference_time)/np.timedelta64(1,'s')
dRsysRadar = -(r_sys.expansion_time - reference_time)/np.timedelta64(1,'s')
s0 = C.ds(start_seconds)

sampleS = [s0 - dRsysRadar + dS*k for k in np.arange(nSamples)]

def closestRoot(root, val):
    return root[np.argmin(np.abs(root - val))]

newTimes = np.zeros((nSamples), dtype=float)
myval = start_seconds
for k,s in zip(range(nSamples), sampleS):
    root = np.roots([tdf[3]/6.0, tdf[2]/2.0, tdf[1],tdf[0]-s])
    myval = closestRoot(root, myval)
    newTimes[k] = myval
    
#%% Compute the ranges to the target
pointXYZ = r_sys.target_ground_position
ref = rd['acquisition']
tmp = np.matlib.repmat(pointXYZ,ref['numAzimuthSamples'],1)
rangeVectors = ref['satellitePositions'][1][:,0:3]-tmp
ranges = np.sqrt(np.sum(rangeVectors*rangeVectors, axis=1))
velocityVectors = ref['satellitePositions'][1][:,3:]
velocityMagnitudes = np.sqrt(np.sum(velocityVectors*velocityVectors, axis=1))
lookDirections = np.sum(rangeVectors*velocityVectors, axis=1)/ranges/velocityMagnitudes
r2t = 2.0/c
rangeTimes = ranges*r2t

#%% Do the same computation but with the expansion
cdf = r_sys.C.cdf
r = np.linalg.norm(r_sys.C.R)
s = np.array(sampleS) - 0.6

# Compute the range curve
sat_full = np.outer(cdf[0], np.ones_like(s)) + np.outer(cdf[1],s) + np.outer(cdf[2], (s**2)/2.0) + np.outer(cdf[3], (s**3)/6.0)
rngs_curve = np.outer(r_sys.C.R, s**0) + np.outer(cdf[1],s) + np.outer(cdf[2], (s**2)/2.0) + np.outer(cdf[3], (s**3)/6.0)
rngs_full = np.sqrt(np.sum(rngs_curve*rngs_curve, axis=0))
rngs_approx = np.sqrt(r**2 + r_sys.C.a2*s**2 + r_sys.C.a3*s**3 + r_sys.C.a4*s**4)
sx = s - 0.39198 #0*r_sys.sx
rngs_approx = np.sqrt(r**2 + 
                      r_sys.C.a2*sx**2 + 
                      r_sys.C.a3*sx**3 + 
                      r_sys.C.a4*sx**4)

#%%
plt.figure()
plt.plot(s,ranges-rngs_approx)
plt.grid()
plt.show()