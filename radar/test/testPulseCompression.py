# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:09:59 2019

@author: SIKANETAI
"""
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

#%%
xmlfile = "E:\\Users\\SIKANETAI\\simulation\\30cm\\reduced_30cm_simulation.xml"
radar = cfg.loadConfiguration(xmlfile)

#%% Get the target point
pointXYZ, satSV, satTime = cfg.computeReferenceGroundPoint(radar, 
                                                      radarIDX = None, 
                                                      rTargetIndex = 400, 
                                                      sTargetIndex = None)

#%% Fix the radar file name
for rad in radar:
    rad["filename"] = "E:" + rad["filename"]
    
#%%
radar_IDX = 18
ref = radar[radar_IDX]["acquisition"]
zpComp = np.load(radar[radar_IDX]["filename"])
nrng, nazi = zpComp.shape
#plt.figure()
#plt.imshow(np.abs(zpComp))
#plt.show()


#
satXYZ = satSV[0:3]
satvXvYvZ = satSV[3:]


#%% Compute some parameters
tmp = np.matlib.repmat(pointXYZ,ref['numAzimuthSamples'],1)
rangeVectors = ref['satellitePositions'][1][:,0:3]-tmp
ranges = np.sqrt(np.sum(rangeVectors*rangeVectors, axis=1))
r2t = 2.0/cfg.physical.c
range_IDX = np.round((ranges*r2t - ref["nearRangeTime"])/ref["rangeSampleSpacing"])
idx = range_IDX<nrng
data_IDX = np.argmax(np.abs(zpComp), axis=0)

#
plt.figure()
plt.plot(data_IDX[idx] - range_IDX[idx], '.-')
plt.grid()
#plt.plot(range(nazi), data_IDX, 'o', range(nazi), range_IDX, '.')
plt.show()

#%%
#velocityVectors = ref['satellitePositions'][1][:,3:]
#velocityMagnitudes = np.sqrt(np.sum(velocityVectors*velocityVectors, axis=1))
#u = np.sum(rangeVectors*velocityVectors, axis=1)/ranges/velocityMagnitudes
##idx = np.abs(u-rad['mode']['txuZero'])<beamwidth/2.0
#
#nRt = ref['nearRangeTime']
#fastTime = ref['nearRangeTime'] + np.arange(float(ref['numRangeSamples']))*ref['rangeSampleSpacing']
#
#Ztx=None
#Zrx=None
#z, Ztx, Zrx = cfg.twoWayArrayPatternTrueTimeFlat(fastTime, u, ranges, rad, Ztx=Ztx, Zrx=Zrx)
#
## Mix the signal to baseband
#z = z*np.matlib.repmat(np.exp(-1j*2.0*np.pi*rad['antenna']['fc']*fastTime),len(ranges),1).T
#
## Pulse compress the signal
#chirp = cfg.chirpWaveform(rad['chirp']['pulseBandwidth'], rad['chirp']['length'])
#S = np.matlib.repmat(np.fft.fft(chirp.sample(fastTime - rad['acquisition']['nearRangeTime'])), len(ranges),1).T
#
## Determine the number of range samples to write to file
#compressed_rs = compressed_range_samples or len(fastTime)
#
## Sanity check for number of samples to write
#compressed_rs = np.max([0, compressed_rs])
#compressed_rs = np.min([len(fastTime), compressed_rs])
#
## Compute the pulse compressed signal
#zpComp = np.fft.ifft(np.fft.fft(z, axis=0)*S.conj(), axis=0)[0:compressed_rs,:]