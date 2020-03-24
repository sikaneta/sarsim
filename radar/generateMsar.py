# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:11:41 2019

@author: SIKANETAI
"""

import sys
import configuration.configuration as cfg
import numpy as np
from measurement.measurement import state_vector
from measurement.arclength import slow
import argparse
import xml.etree.ElementTree as etree
import os
from numba import cuda
from scipy.constants import Boltzmann
if cuda.is_available():
    from antenna.pattern import antennaResponseCudaMem as antennaResp
else:
    from antenna.pattern import antennaResponseMultiCPU as antennaResp
    
#%% Load the data
parser = argparse.ArgumentParser(description="Generate simulated SAR data")

parser.add_argument("--config-xml", help="The config XML file", 
                    default = u'/home/ishuwa/simulation/40cm/40cm_simulation.xml')
parser.add_argument("--radar-idx",
                    help=""" The indexes of the radar files to create. This 
                    allows the program to be run simulataneously on different 
                    CPUs to create the data""",
                    nargs = "*",
                    type=int,
                    default = None)
# parser.add_argument("--compressed-range-samples",
#                     help="The number of compressed range samples to retain in each file",
#                     type=int,
#                     default = None)
vv = parser.parse_args()

#%%
radar = cfg.loadConfiguration(vv.config_xml)

#%% Generate a reference ground point
rTargetIndex = 400
pointXYZ, satSV, satTime = cfg.computeReferenceGroundPoint(radar, 
                                                      None, 
                                                      400, 
                                                      None)  
satXYZ = satSV[0:3]
satvXvYvZ = satSV[3:]  

#%% Compute the signals
def computeSignal(radar, pointXYZ, satSV): 
    
    # Loop through radars and compute and write data
    for rad in radar:
        ref = rad['acquisition']
        print("Computing file: %s" % os.path.split(rad['filename'])[-1])
        print("="*80)
        
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
        
        #% Define the fast time values
        fastTimes = ref['nearRangeTime'] + np.arange(float(ref['numRangeSamples']))*ref['rangeSampleSpacing']
        
        #% Compute the antenna pattern amplitudes and delays
        azimuthPositions = rad['antenna']['azimuthPositions']/cfg.physical.c
        txMag = rad['mode']['txMagnitude']
        rxMag = rad['mode']['rxMagnitude']
        txDelay = rad['mode']['txDelay']
        rxDelay = rad['mode']['rxDelay']
        
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
                    *np.sqrt(np.sum(rad['antenna']['transmitPowers'][txMag>0])/(4.0*np.pi)**3)
                    /np.sqrt(Boltzmann*rad['antenna']['systemTemperature'])
                    /np.sqrt(rad['antenna']['systemLosses']))/(fastTimes/r2t)**2
        
        # Define the output array
        pulse_data = np.zeros((len(fastTimes), len(ranges)), dtype=np.complex128)
        for pulseIDX in range(len(ranges)):
            if np.mod(pulseIDX, 100) == 0:
                print("Progress %0.4f percent" % (pulseIDX/len(ranges)*100.0))
            pulse_data[:,(pulseIDX+0)] = antennaResp(fastTimes, 
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
        
        # Get the target domain from the filename
        domain = rad['filename'].split("_")[-2]
        
        fn_dict = {'rx': lambda x: x, 
                   'Rx': lambda x: np.fft.fft(x, axis=0),
                   'rX': lambda x: np.fft.fft(x, axis=1),
                   'RX': lambda x: np.fft.fft2(x)}
        
        # Choose the domain for the data written to file
        pulse_data = fn_dict[domain](pulse_data)
        
        # Write the file to disk
        np.save(rad['filename'], pulse_data)
        

#%% New generation of data
computeSignal(radar, pointXYZ, satXYZ)       