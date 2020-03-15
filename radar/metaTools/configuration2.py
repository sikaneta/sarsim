#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:09:17 2018

@author: sikaneta
"""

radar = loadConfiguration()

#%% Yeah, do some stuff
pointXYZ, satXYZ = computeReferenceGroundPoint(radar)
beamwidth = radar[11]['mode']['txuZero'] - radar[10]['mode']['txuZero']
rad = radar[0]
ref = rad['acquisition']

tmp = np.matlib.repmat(pointXYZ,ref['numAzimuthSamples'],1)
rangeVectors = ref['satellitePositions'][1][:,0:3]-tmp
ranges = np.sqrt(np.sum(rangeVectors*rangeVectors, axis=1))
velocityVectors = ref['satellitePositions'][1][:,3:]
velocityMagnitudes = np.sqrt(np.sum(velocityVectors*velocityVectors, axis=1))
u = np.sum(rangeVectors*velocityVectors, axis=1)/ranges/velocityMagnitudes
idx = np.abs(u-rad['mode']['txuZero'])<beamwidth/2.0
r2t = 2.0/physical.c
nRt = ref['nearRangeTime']
rangeSpan = (min(ranges[idx])*r2t - nRt)/ref['rangeSampleSpacing'], (max(ranges[idx])*r2t - nRt)/ref['rangeSampleSpacing']
nChirp = rad['chirp']['length']/ref['rangeSampleSpacing']
rngSampleMargin = 100
numRangeSamples = np.ceil(rangeSpan[1] - rangeSpan[0] + nChirp) + rngSampleMargin
# Calculate a new nearRange time
nearRangeTime = ref['nearRangeTime'] + (int((min(ranges[idx])*r2t - ref['nearRangeTime'])/ref['rangeSampleSpacing']) - rngSampleMargin)*ref['rangeSampleSpacing']
ref['nearRangeTime'] = nearRangeTime
ref['numRangeSamples'] = numRangeSamples
fastTime = ref['nearRangeTime'] + np.arange(float(ref['numRangeSamples']))*ref['rangeSampleSpacing']
Ztx=None
Zrx=None
z, Ztx, Zrx = twoWayArrayPatternTrueTime(fastTime, u, ranges, rad, Ztx=Ztx, Zrx=Zrx)
plt.imshow(np.abs(z))