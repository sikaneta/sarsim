# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:38:06 2019

@author: SIKANETAI
"""

#%%
""" This python program should generate the config files for burst mode multi or single channel data
    The program generates an xml file to be processed in MATLAB """

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
from scipy.constants import Boltzmann
from antenna.pattern import antennaResponseMultiCPU as antennaResp
from numba import cuda
if cuda.is_available():
    from antenna.pattern import antennaResponseCudaMem as antennaResp
import time

#%%
resolution = 1.0
res_str = "%dcm" % int(resolution*100) 

#%% Load the configuration
#radar = cfg.loadConfiguration("E:\\Users\\SIKANETAI\\simulation\\30cm\\reduced_30cm_simulation.xml")
radar = cfg.loadConfiguration("/home/ishuwa/simulation/40cm/reduced_40cm_simulation.xml")

#%%
pointXYZ, satSV, satTime = cfg.computeReferenceGroundPoint(radar, None, 400, None)
satXYZ = satSV[0:3]
satvXvYvZ = satSV[3:]
    
#%%
rad = radar[0]
ref = rad['acquisition']

# C = slow(ref['satellitePositions'][0])

# Compute some parameters
tmp = np.matlib.repmat(pointXYZ,ref['numAzimuthSamples'],1)
rangeVectors = ref['satellitePositions'][1][:,0:3]-tmp
ranges = np.sqrt(np.sum(rangeVectors*rangeVectors, axis=1))
velocityVectors = ref['satellitePositions'][1][:,3:]
velocityMagnitudes = np.sqrt(np.sum(velocityVectors*velocityVectors, axis=1))
lookDirections = np.sum(rangeVectors*velocityVectors, axis=1)/ranges/velocityMagnitudes
#idx = np.abs(u-rad['mode']['txuZero'])<beamwidth/2.0
r2t = 2.0/cfg.physical.c
rangeTimes = ranges*r2t
nRt = ref['nearRangeTime']

#%% Define the fast time values
fastTimes = ref['nearRangeTime'] + np.arange(float(ref['numRangeSamples']))*ref['rangeSampleSpacing']

#%%
#pulseIDX = 7381
#fastTimes = fastTime
#targetRangeTime =rangeTimes[pulseIDX]
#u = lookDirections[pulseIDX]
#minAntennaLength = np.min(rad['antenna']['azimuthLengths'])

#%%
azimuthPositions = rad['antenna']['azimuthPositions']/cfg.physical.c
txMag = rad['mode']['txMagnitude']
rxMag = rad['mode']['rxMagnitude']
txDelay = rad['mode']['txDelay']
rxDelay = rad['mode']['rxDelay']
#chirp_bandwidth = rad['chirp']['pulseBandwidth']
#chirp_duration = rad['chirp']['length']
#chirp_carrier = rad['antenna']['fc']

myTxAmp = txMag[txMag > 0.0]
myTxAvg = myTxAmp/np.sum(myTxAmp)
myTxDly = txDelay[txMag > 0.0]
myTxPos = azimuthPositions[txMag > 0.0]
myTxLen = myTxAmp.shape[0]

myRxAmp = rxMag[rxMag > 0.0]
myRxAvg = myRxAmp/np.sum(myRxAmp)
myRxDly = rxDelay[rxMag > 0.0]
myRxPos = azimuthPositions[rxMag > 0.0]
myRxLen = myRxAmp.shape[0]

radarEq1 = np.sqrt((rad['antenna']['wavelength'])
                    *np.sqrt(np.sum(rad['antenna']['transmitPowers'][txMag>0]))
                    /(4.0*np.pi)
                    /np.sqrt(Boltzmann*rad['antenna']['systemTemperature'])
                    /np.sqrt(rad['antenna']['systemLosses']))/(fastTimes/r2t)**2

pattern_delay = np.dot(myTxDly, myTxAvg) + np.dot(myRxDly, myRxAvg)
dly = [np.dot(u*myTxPos, myTxAvg) + np.dot(u*myRxPos, myRxAvg) for u in lookDirections]
dlySmp = [d/ref['rangeSampleSpacing'] for d in dly]

#%%
pulseIDX = 2369
raw_data = np.zeros((len(fastTimes), len(ranges)), dtype=np.complex128)
t1 = time.time()
for pulseIDX in range(len(ranges)):
    if np.mod(pulseIDX, 100) == 0:
        print("Progress %0.4f percent" % (pulseIDX/len(ranges)*100.0))
#    t1 = time.time()
#    raw_data[:,pulseIDX] = antennaResponse(fastTimes, 
#                           rangeTimes[pulseIDX],
#                           lookDirections[pulseIDX],
#                           np.min(rad['antenna']['azimuthLengths']),
#                           rad['antenna']['azimuthPositions']/cfg.physical.c,
#                           rad['mode']['txMagnitude'],
#                           rad['mode']['rxMagnitude'],
#                           rad['mode']['txDelay'],
#                           rad['mode']['rxDelay'],
#                           rad['chirp']['pulseBandwidth'],
#                           rad['chirp']['length'],
#                           rad['antenna']['fc'])
#    t2 = time.time()
    raw_data[:,(pulseIDX+0)] = antennaResp(fastTimes, 
                           rangeTimes[pulseIDX],
                           lookDirections[pulseIDX],
                           np.min(rad['antenna']['azimuthLengths']),
                           rad['antenna']['azimuthPositions']/cfg.physical.c,
                           rad['mode']['txMagnitude'],
                           rad['mode']['rxMagnitude'],
                           rad['mode']['txDelay'],
                           rad['mode']['rxDelay'],
                           rad['chirp']['pulseBandwidth'],
                           rad['chirp']['length'],
                           rad['antenna']['fc'])*radarEq1
#    t3 = time.time()
#    print(t2 - t1)
#    print(t3 - t2)
t2 = time.time()
print(t2 - t1)
#%%
plt.figure()
plt.plot(np.abs(np.sum(raw_data, axis=0)))
plt.show()    

#%%
plt.figure()
plt.plot(np.abs(raw_data[:,pulseIDX]))
plt.grid()
plt.show()

plt.figure()
plt.plot(np.abs(raw_data[:,(pulseIDX+1)]))
plt.grid()
plt.show()

#%% Compute some parameters
range_IDX = (ranges*r2t - ref["nearRangeTime"])/ref["rangeSampleSpacing"]
idx = range_IDX<raw_data.shape[0]
data_IDX = np.argmax(np.abs(raw_data), axis=0)

plt_DATA = data_IDX[idx]
plt_RNG = range_IDX[idx]

#
plt.figure()
plt.plot(range(len(plt_DATA)), plt_DATA, range(len(plt_DATA)), plt_RNG, '.')
plt.grid()
#plt.plot(range(nazi), data_IDX, 'o', range(nazi), range_IDX, '.')
plt.show()

#%%
rdata = cfg.loadNumpy_raw_data(radar[0:6], target_domain="rx")

#%%
H = np.array([cfg.twoWayArrayPatternLinearKS(r_sys.ks, rd, r_sys.kr[0], ds) 
                  for rd,ds in zip(radar,r_sys.s_offsets)])
    
#%%
c0 = 0
c1 = 4
dintf = np.sum(rdata[c0,:,:], axis=0)*np.conj(np.sum(rdata[c1,:,:], axis=0))

#
x = np.arange(len(dintf))/r_sys.ksp
x = x - np.mean(x)
plt.figure()
plt.plot(x, np.abs(dintf),'.')
plt.grid()
plt.show()

#
plt.figure()
plt.plot(x, np.angle(dintf*np.exp(1j*r_sys.kr[0]*(-(c1-c0)*1.8)*lookDirections)),'.')
plt.grid()
plt.show()

#
plt.figure()
h1 = H[c0,:,:].flatten()
h2 = H[c1,:,:].flatten()
intf = h1*np.conj(h2)*np.exp(-1j*(-(c1-c0)*1.8)*r_sys.ks.flatten())
plt.plot(r_sys.ks.flatten(), np.angle(intf),'.')
plt.ylim((-np.pi, np.pi))
plt.grid()
plt.show()











