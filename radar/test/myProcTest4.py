# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:57:13 2019

@author: SIKANETAI
"""

import configuration.configuration as cfg
import numpy as np
import matplotlib.pyplot as plt
from measurement.measurement import state_vector
from measurement.arclength import slow
import argparse
from measurement.measurement import state_vector
import os
#%matplotlib notebook

#%%
#pth = "/home/ishuwa_tinda/local/src/Python/radarAugust2019/radar/radar" 
#pth = "E:\\Python\\myrdr4\\radar" 
pth = '/home/ishuwa/local/src/Python/radarAugust222019/myrdr4/radar'
#xmlFile = "sureConfighalfhalf.xml"
xmlFile = "reduced_sureConfig1M1.xml"

#%% Load the radar configuration
radar = cfg.loadConfiguration(os.path.join(os.path.join(pth, "XML"),xmlFile))
#radar = cfg.loadConfiguration("E:\\Python\\myrdr2\\radar\\XML\\sureTest1M.xml")

#%%
fls = [os.path.join(pth, r['filename']) for r in radar]
domain = [fl.split("_")[-2] for fl in fls]
print(fls)

#%% Define the data loading function dictionary
fn_dict = {'rx': lambda x: np.fft.fft(x, axis=1), 
           'Rx': lambda x: np.fft.fft(x, axis=1),
           'rX': lambda x: x,
           'RX': lambda x: x}

#%% Load the data
#data = np.stack([np.fft.fft(np.load(fl), axis=1) for fl in fls], axis=0)
data = np.stack([fn_dict[dm](np.load(fl)) for dm,fl in zip(domain,fls)], axis=0)

#%% Define which bands to look at
bands = np.arange(-6,7)
bands = np.arange(-int(len(radar)/2)-2,int(len(radar)/2)+1+2)

#%% Define the arrays
rad = radar[int(len(radar)/2)]
f0 = rad['antenna']['fc']
ref = rad['acquisition']
fs = 1.0/ref['rangeSampleSpacing']
Nr = ref['numRangeSamples']
Na = ref['numAzimuthSamples']
prf = ref['prf']
nearRangeTime = rad['acquisition']['nearRangeTime']

cSat = int(Na/2)
sv = state_vector()
xState = sv.expandedState(ref['satellitePositions'][1][cSat], 0.0)
print(xState)
satXYZ = xState[0,:]
print(satXYZ)
satvXvYvZ = xState[1,:]
C = slow(ref['satellitePositions'][0])
cdf, tdf, T, N, B, kappa, tau, dkappa = C.diffG(xState)
C.t2s()

#%% Generate a reference ground point
cAntenna = int(len(radar)/2)
refTimePos = radar[cAntenna]['acquisition']['satellitePositions']
rfIDX = int(len(refTimePos[0])/2)

pointXYZ, satSV = cfg.computeReferenceGroundPoint(radar, 
                                                  radarIDX = cAntenna, 
                                                  rTargetIndex = 400, 
                                                  sTargetIndex = rfIDX)
rvec = satXYZ-pointXYZ
a2 = 1.0 - C.kappa*np.dot(-rvec, C.N)
vs = np.linalg.norm(satvXvYvZ)
print("Vsat: %0.4f" % vs)

ksp = prf/vs
ksp = 1.0/(C.ds(1.0/prf))
kr = 4.0*np.pi*cfg.FFT_freq(Nr, fs, 0.0)/cfg.physical.c + 4.0*np.pi*f0/cfg.physical.c

ks = np.array([2.0*np.pi*cfg.FFT_freq(Na, ksp, b*ksp) for b in bands])
ksidx = np.argsort(ks.flatten())

#%%Calculate spatial offsets from motion
s_offsets = [C.ds((r['acquisition']['satellitePositions'][0][0] 
              - radar[0]['acquisition']['satellitePositions'][0][0]).total_seconds()) for r in radar]
#s_offsets = np.array([0.0,0.0,0.0,1.0,1.0,1.0,2.0,2.0,2.0])
print("Sampling offsets:")
print(s_offsets)

#%% Generate the antenna patterns for all channels
H = np.array([cfg.twoWayArrayPatternLinearKS(ks, rd, kr[0], ds) for rd,ds in zip(radar,s_offsets)])
print(H.shape)
print(data.shape)

#%% Generate the desired "final" antenna pattern
D = np.sqrt(np.sum(np.abs(H)**2, axis=0))

#%%
plt.figure()
plt.plot(D.flatten()[ksidx])
plt.show()

#%% Use a noise matrix
Rn = np.eye(len(radar))

#%% Look at the filter
myH = np.sum(H[:,:,:],axis=1)
myH = myH/np.max(np.abs(myH.flatten()))

#%% Look at the data
myD = np.sum(data,axis=1)
myD = myD/np.max(np.abs(myD))

#%%
plt.figure()
plt.plot(ks[8,:],np.angle(myH.T),'.')
plt.grid()
plt.show()

#%% Some plots
ix0 = 3
ix1 = 4
myintD = myD[ix0,:]*np.conj(myD[ix1,:])
myintH = myH[ix0,:]*np.conj(myH[ix1,:])
print(myintD.shape)
print(myintH.shape)
plt.figure()
plt.plot(ks[2,:],np.angle(myintD),'.', ks[2,:],np.angle(myintH),'.')
plt.show()

#%% Compute the inverse matrices
Nb = len(bands)
p = 0.5
procData = np.zeros((Nb,Na,Nr), dtype=np.complex128)
for kidx in range(Na):
    if kidx%300 == 1:
        print("Progess: %0.2f" % (100.0*kidx/Na))
    Rinv = np.linalg.inv(np.dot(H[:,:,kidx], np.conj(H[:,:,kidx].T)) + (1.0-p)/p*Rn)
    B = np.dot(np.diag(D[:,kidx]), np.dot(np.conj(H[:,:,kidx].T), Rinv))
    if np.any(np.isnan(B)):
        print(kidx)
        break
    dummy = np.dot(B,data[:,:,kidx])
    for bidx in range(Nb):
        procData[bidx,kidx,:] = dummy[bidx,:]

#%% Release data from memory
del data
        
#%% Reorder the data
ks_full = 2.0*np.pi*cfg.FFT_freq(Na*Nb, ksp*Nb, 0)
ks_full_idx = np.argsort(ks_full) - (2 - len(ks_full)%2)
procData = procData.reshape((Nb*Na, Nr))
procData = procData[ksidx[ks_full_idx],:]

#%% Look at a plot
plt.figure()
plt.figure()
plt.plot(np.abs(np.sum(procData,axis=1)),'.')
plt.show()

#%%
procData = np.fft.ifft(procData, axis=0)

#%%
flatProc = np.sum(procData, axis=1)

#%%
plt.figure()
plt.grid()
plt.plot(np.abs(flatProc))
plt.show()

#%%
cAntenna = int(len(radar)/2)
refTimePos = radar[cAntenna]['acquisition']['satellitePositions']
rfIDX = int(len(refTimePos[0])/2)

pointXYZ, satSV = cfg.computeReferenceGroundPoint(radar, 
                                                  radarIDX = cAntenna, 
                                                  rTargetIndex = 400, 
                                                  sTargetIndex = rfIDX)

#%%
s = np.arange(Na*Nb)/(ksp*Nb)
#s = np.arange(Na)/ksp
#slow_t = np.arange(Na)/prf
#slow_t -= np.mean(slow_t)
s = s - np.mean(s) - 5.27
#s.shape
#s = C.s + 10.5
print(cdf)
rngs_curve = np.outer(cdf[0] - pointXYZ, s**0) + np.outer(cdf[1],s) + np.outer(cdf[2], (s**2)/2.0) + np.outer(cdf[3], (s**3)/6.0)
rngs = np.sqrt(np.sum(rngs_curve*rngs_curve, axis=0))
#rngs = sat_curve - np.repmat(pointXYZ, (1,Na*Nb))
rC = np.exp(-1j*kr[0]*rngs)

#%%
plt.figure()
plt.plot(np.unwrap(np.angle(flatProc*np.conj(rC))),'.')
plt.grid()
plt.show()