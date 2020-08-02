#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:38:54 2020

@author: ishuwa
"""

import configuration.configuration as cfg
import plot.simplot as sp
import numpy as np
from scipy.signal import hann as weightwin
from scipy.constants import c, Boltzmann
import argparse
import os
import utils.fileio as fio
from tqdm import tqdm

#%% Argparse stuff
purpose = """
          Plot some figures from the SAR processed
          data. These can be used to check the quality of the
          processing
          """
parser = argparse.ArgumentParser(description=purpose)

parser.add_argument("--config-xml", 
                    help="The config XML file", 
                    required=True)
parser.add_argument("--phase-correct",
                    help="""Do not remove residual phase
                            This is the phase between the expansion of the
                            satellite position using differential geometry
                            and the numerically integrated satellite
                            position""",
                    action="store_true",
                    default=False)
parser.add_argument("--power-correct",
                    help="""Correct error in power computation.
                            Simulations run after 31/07/2020 should
                            not need this correction""",
                    action="store_true",
                    default=False)
parser.add_argument("--arclength-offset",
                    help="Offset of the target from zero in arclength (m)",
                    type=float,
                    default=0.0)
parser.add_argument("--elevation-length",
                    help="Length of the antenna in the elevation direction",
                    type=None,
                    default=None)
parser.add_argument("--ridx",
                    help="The range indeces to examine",
                    type=int,
                    nargs=2,
                    default=[0,None])
parser.add_argument("--xidx",
                    help="The azimuth indeces to examine",
                    type=int,
                    nargs=2,
                    default=[0,None])
parser.add_argument("--interactive",
                    help="Interactive mode. Program halts until figures are closed",
                    action="store_true",
                    default=False)

#%% Parse the arguments           
vv = parser.parse_args()

#%% Load the radar configuration
if 'radar' not in locals():
    radar = cfg.loadConfiguration(vv.config_xml)
    
#%% Calculate noise and get an r_sys object
noiseSignal, r_sys = cfg.multiChannelNoise(radar, p=0.9)
noiseSignal = np.fft.ifft(noiseSignal)
noisePower = np.var(noiseSignal)
correction = 1.0

if vv.power_correct:
    rad = radar[0]
    txMag = rad['mode']['txMagnitude']
    radarEq1 = np.sqrt((rad['antenna']['wavelength'])
                        *np.sqrt(np.sum(rad['antenna']['transmitPowers'][txMag>0])/(4.0*np.pi)**3)
                        /np.sqrt(Boltzmann*rad['antenna']['systemTemperature'])
                        /np.sqrt(rad['antenna']['systemLosses']))
            
            
    radarEq2 = ((rad['antenna']['wavelength'])
                *np.sqrt(np.sum(rad['antenna']['transmitPowers'][txMag>0])/(4.0*np.pi)**3)
                /np.sqrt(Boltzmann*rad['antenna']['systemTemperature']*rad['chirp']['pulseBandwidth'])
                /np.sqrt(rad['antenna']['systemLosses'])
                )
    
    correction = (radarEq2/radarEq1)**2


#%% Load the data
if 'wkSignal'not in locals():
    proc_file = fio.fileStruct(radar[0]['filename'],
                                        "wk_processed",
                                        "Xr",
                                        "wkprocessed.npy")
    print("Loading data from file: %s" % proc_file)
    wkSignal = fio.loadSimFiles(proc_file, xidx=vv.xidx, ridx=vv.ridx)
    
    # Apply the phase correction
    rows, cols = wkSignal.shape
    if vv.phase_correct:
        print("Removing residual phase")
        for k in tqdm(range(rows)):
            wkSignal[k,:] *= np.exp(-1j*r_sys.ks_phase_correction[k])
    
    
    # Shift the signal as required
    print("Attempting to shift the signal...")
    s = np.arange(r_sys.Na*r_sys.n_bands)/(r_sys.ksp*r_sys.n_bands)
    s -= np.mean(s)
    mxcol = np.argmax(wkSignal[0,:])
    intf = wkSignal[0:-1, mxcol]*np.conj(wkSignal[1:, mxcol])
    intf_weight = np.fft.fftshift(weightwin(len(intf)))**6
    dks = r_sys.ks_full[1] - r_sys.ks_full[0]
    c_ang = np.angle(np.sum(intf*intf_weight)) - dks*np.min(s)
    s_off = c_ang/dks
    p_fct = np.exp(1j*r_sys.ks_full*s_off)
    for k in tqdm(np.arange(cols)):
        wkSignal[:, k] *= p_fct
    print("Computing the FFT of the signal ...")
    wkSignal = np.fft.ifft(wkSignal, axis=0)
    maxSignal = np.max(np.abs(wkSignal))**2*correction
    wkSignal = wkSignal/maxSignal
    
    # Use some estimate for the antenna element gain 4*pi/lambda**2*Area
    eLen = vv.elevation_length or radar[0]['antenna']['elevationLengths'][0]
    aLen = radar[0]['antenna']['azimuthLengths'][0]
    
    elementGainTx = 10*np.log10(2*np.sqrt(np.pi)/(c/r_sys.f0)*eLen)
    elementGainTx += 10*np.log10(2*np.sqrt(np.pi)/(c/r_sys.f0)*aLen)
    elementGainRx = 10*np.log10(2*np.sqrt(np.pi)/(c/r_sys.f0)*eLen)
    elementGainRx += 10*np.log10(2*np.sqrt(np.pi)/(c/r_sys.f0)*aLen)
    NESZ = (-10*np.log10(maxSignal/noisePower) 
            - elementGainTx 
            - elementGainRx)
    print('NESZ: %0.2f dB' % NESZ)



#%% Define the folder in which to store plots
sim_folder = os.path.join(os.path.split(vv.config_xml)[0], 
                          "simulation_plots")

#%% Do the plotting
sp.sarprocPlot(wkSignal, 
               r_sys, 
               interactive=vv.interactive, 
               folder=sim_folder, 
               s_off=vv.arclength_offset)
