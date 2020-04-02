# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:11:41 2019

@author: SIKANETAI
"""

import configuration.configuration as cfg
import utils.fileio as fio
import numpy as np
import argparse
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
                    default = u'/home/ishuwa/simulation/40cm/simulation_40cm.xml')
parser.add_argument("--radar-idx",
                    help=""" The indexes of the radar files to create. This 
                    allows the program to be run simulataneously on different 
                    CPUs to create the data""",
                    nargs = "*",
                    type=int,
                    default = None)
parser.add_argument("--range-block",
                    help="""The output files for each beam/channel will be
                            written to files with the range dimension of
                            size range block. 
                            i.e. with 
                            - a range block size of --range-block=512 
                            - a data file with name (...rX_c0b0.npy)
                            - a data size of 1200X16384 in range by
                              azimuth respectively (rX) 
                            then will write several files of names 
                            ...r0X0_c0b0.npy    (has 512 rows)
                            ...r512X0_c0b0.npy  (has 512 rows)
                            ...r1024X0_c0b0.npy (has 1200-1024 rows)
                            If not specified, will set block to the total
                            number of range samples.""",
                    type=int,
                    default = None)
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
def computeSignal(radar, pointXYZ, satSV, range_block=None): 
    
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
        r2t = 2.0/cfg.physical.c
        rangeTimes = ranges*r2t
        
        #% Define the fast time values
        fastTimes = ref['nearRangeTime'] + np.arange(float(ref['numRangeSamples']))*ref['rangeSampleSpacing']
        
        #% Compute the antenna pattern amplitudes and delays
        txMag = rad['mode']['txMagnitude']
        
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
        filenames = fio.writeSimFiles(rad['filename'], 
                                      pulse_data,
                                      rblock = range_block)
        #np.save(rad['filename'], pulse_data)
        

#%% New generation of data
computeSignal(radar, pointXYZ, satXYZ, vv.range_block)       