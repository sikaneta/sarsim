# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:11:41 2019

@author: SIKANETAI
"""
#cd "e:\\Python\\myrdr2\\radar"

import sys
import configuration.configuration as cfg
import numpy as np
from measurement.measurement import state_vector
from measurement.arclength import slow
import argparse
import lxml.etree as etree
import os
from numba import cuda
if cuda.is_available():
    from antenna.pattern import antennaResponseCudaMem as antennaResp
else:
    from antenna.pattern import antennaResponseMultiCPU as antennaResp
    
#%% Define a dictionary of functions for transforming the data
fn_dict = {'rx': lambda x: x, 
           'Rx': lambda x: np.fft.fft(x, axis=0),
           'rX': lambda x: np.fft.fft(x, axis=1),
           'RX': lambda x: np.fft.fft2(x)}

#%% Define a function to compute the signal for a given radar object, a
# target position and the satellite position
def computeRadarSignalGPU(rad, pointXYZ, satSV):
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
    
    pattern_delay = np.dot(myTxDly, myTxAvg) + np.dot(myRxDly, myRxAvg)
    
    # Define the output array
    pulse_data = np.zeros((len(fastTimes), len(ranges)), dtype=np.complex128)
    for pulseIDX in range(len(ranges)):
        if np.mod(pulseIDX, 100) == 0:
            print("Progress %0.4f percent" % (pulseIDX/len(ranges)*100.0))
        pulse_data[:,(pulseIDX+0)] = antennaResp(fastTimes + pattern_delay, 
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
                               rad['antenna']['fc'])
    
    # Get the target domain from the filename
    domain = rad['filename'].split("_")[-2]
    
    # Choose the domain for the data written to file
    return fn_dict[domain](pulse_data)
        
#%% Compute the signal for a single channel/beam
def computeRadarSignal(rad, 
                       pointXYZ,
                       satXYZ,
                       compressed_range_samples):
    ref = rad['acquisition']
    
    # Compute some parameters
    tmp = np.matlib.repmat(pointXYZ,ref['numAzimuthSamples'],1)
    rangeVectors = ref['satellitePositions'][1][:,0:3]-tmp
    ranges = np.sqrt(np.sum(rangeVectors*rangeVectors, axis=1))
    velocityVectors = ref['satellitePositions'][1][:,3:]
    velocityMagnitudes = np.sqrt(np.sum(velocityVectors*velocityVectors, axis=1))
    u = np.sum(rangeVectors*velocityVectors, axis=1)/ranges/velocityMagnitudes
    #idx = np.abs(u-rad['mode']['txuZero'])<beamwidth/2.0
    r2t = 2.0/cfg.physical.c
    nRt = ref['nearRangeTime']
    fastTime = ref['nearRangeTime'] + np.arange(float(ref['numRangeSamples']))*ref['rangeSampleSpacing']
    
    Ztx=None
    Zrx=None
    z, Ztx, Zrx = cfg.twoWayArrayPatternTrueTimeFlat(fastTime, u, ranges, rad, Ztx=Ztx, Zrx=Zrx)
    
    # Mix the signal to baseband
    z = z*np.matlib.repmat(np.exp(-1j*2.0*np.pi*rad['antenna']['fc']*fastTime),len(ranges),1).T
    
    # Pulse compress the signal
    chirp = cfg.chirpWaveform(rad['chirp']['pulseBandwidth'], rad['chirp']['length'])
    S = np.matlib.repmat(np.fft.fft(chirp.sample(fastTime - rad['acquisition']['nearRangeTime'])), len(ranges),1).T
    
    # Determine the number of range samples to write to file
    compressed_rs = compressed_range_samples or len(fastTime)
    
    # Sanity check for number of samples to write
    compressed_rs = np.max([0, compressed_rs])
    compressed_rs = np.min([len(fastTime), compressed_rs])
    
    # Compute the pulse compressed signal
    zpComp = np.fft.ifft(np.fft.fft(z, axis=0)*S.conj(), axis=0)[0:compressed_rs,:]
    
    # Get the target domain from the filename
    domain = rad['filename'].split("_")[-2]
    
    # Choose the domain for the data written to file
    return fn_dict[domain](zpComp)

#%% Compute the signals
def computeSignal(radar, 
                  xml_file,
                  pointXYZ,
                  satXYZ,
                  compressed_range_samples = None):
#                  radarIDX = None, 
#                  rTargetIndex = 400, 
#                  sTargetIndex = None, 
#                  compressed_range_samples = None):
    #% Generate a reference ground point
#    pointXYZ, satSV, satTime = cfg.computeReferenceGroundPoint(radar, 
#                                                      radarIDX, 
#                                                      rTargetIndex, 
#                                                      sTargetIndex)
#    satXYZ = satSV[0:3]
#    satvXvYvZ = satSV[3:]
    
    # Loop through radars and compute and write data
    for rad in radar:
        ref = rad['acquisition']
        # C = slow(ref['satellitePositions'][0])
        
        # Compute some parameters
        tmp = np.matlib.repmat(pointXYZ,ref['numAzimuthSamples'],1)
        rangeVectors = ref['satellitePositions'][1][:,0:3]-tmp
        ranges = np.sqrt(np.sum(rangeVectors*rangeVectors, axis=1))
        velocityVectors = ref['satellitePositions'][1][:,3:]
        velocityMagnitudes = np.sqrt(np.sum(velocityVectors*velocityVectors, axis=1))
        u = np.sum(rangeVectors*velocityVectors, axis=1)/ranges/velocityMagnitudes
        #idx = np.abs(u-rad['mode']['txuZero'])<beamwidth/2.0
        r2t = 2.0/cfg.physical.c
        nRt = ref['nearRangeTime']
        fastTime = ref['nearRangeTime'] + np.arange(float(ref['numRangeSamples']))*ref['rangeSampleSpacing']
        
        Ztx=None
        Zrx=None
        z, Ztx, Zrx = cfg.twoWayArrayPatternTrueTimeFlat(fastTime, u, ranges, rad, Ztx=Ztx, Zrx=Zrx)
        
        # Mix the signal to baseband
        z = z*np.matlib.repmat(np.exp(-1j*2.0*np.pi*rad['antenna']['fc']*fastTime),len(ranges),1).T
        
        # Pulse compress the signal
        chirp = cfg.chirpWaveform(rad['chirp']['pulseBandwidth'], rad['chirp']['length'])
        S = np.matlib.repmat(np.fft.fft(chirp.sample(fastTime - rad['acquisition']['nearRangeTime'])), len(ranges),1).T
        
        # Determine the number of range samples to write to file
        compressed_rs = compressed_range_samples or len(fastTime)
        
        # Sanity check for number of samples to write
        compressed_rs = np.max([0, compressed_rs])
        compressed_rs = np.min([len(fastTime), compressed_rs])
        
        # Compute the pulse compressed signal
        zpComp = np.fft.ifft(np.fft.fft(z, axis=0)*S.conj(), axis=0)[0:compressed_rs,:]
        
        # Get the target domain from the filename
        domain = rad['filename'].split("_")[-2]
        
        # Choose the domain for the data written to file
        zpComp = fn_dict[domain](zpComp)
        
        # Write the file to disk
        np.save(rad['filename'], zpComp)
        
    # Check if we need to write a new config file
    if compressed_rs < len(fastTime):
        head,tail = os.path.split(xml_file)
        newxml = os.path.join(head,"reduced_%s" % tail)
        xmlroot = etree.parse(xml_file).getroot()
        for rS in xmlroot.findall(".//numRangeSamples"):
            rS.text = "%d" % compressed_rs
        
        with open(newxml, 'w') as f:
            #f.write(dom.toprettyxml())
            f.write(etree.tostring(xmlroot, pretty_print=True).decode())
                
    return newxml
            
        