#!/usr/bin/env python2.7

#%%
""" This python program should generate the config files for burst mode multi 
    or single channel data.
    The program generates an xml file to be processed in MATLAB """

import numpy as np
# from datetime import datetime, date, time, timedelta
# from math import pi
# import urllib
# from copy import deepcopy
# import lxml.etree as etree
# from functools import reduce
# import os
from scipy.constants import c
# from generateXML.base_XML import base_ROSEL
import pandas as pd
# import os
import matplotlib.pyplot as plt
from types import SimpleNamespace

from orbit.orientation import orbit as kepler_orbit
from orbit.geometry import getTiming
from measurement.measurement import state_vector

#%% Coefficients file
roseLxsl = r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\RoseL\ROSE-UM-ADSF-SAR-1001590382_Iss.01_Antenna_Model_Installation\ROSEL_AntennaModel\Inputs\ROSEL_DPNear_mid.xlsx"

#%%
class sptRadar:
    antenna = SimpleNamespace()
    orbit = SimpleNamespace(kepler = kepler_orbit(e = 0.001185, 
                                                  arg_perigee = np.radians(90.58), 
                                                  a = 7070946, 
                                                  inclination = np.radians(98.1813)),
                            base_time = np.datetime64("2022-12-06T18:00:00.000000000"),
                            sv = state_vector())
    
    xlsx = SimpleNamespace()
    
    def __init__(self, xslxFile, modeName, swathNum=1):
        self.populateStateVector()
        self.update(xslxFile, modeName)
        self.readChannels()
        
    def populateStateVector(self):
        kepler = self.orbit.kepler
        sv = self.orbit.sv
        base_time = self.orbit.base_time
        t = np.arange(1001)/1001*kepler.period
        llh = []
        for tm in t:
            sv_time = base_time + np.timedelta64(int(tm*1e9), 'ns')
            sv_PCI = kepler.computeSV(kepler.computeO(tm)[0])[0]
            sv_PCR = sv.toPCR(sv_PCI, tm)
            llh.append(sv.xyz2polar(sv_PCR))
            sv.add(sv_time, sv_PCR)
            
        self.orbit.min_height_idx = np.argmin([l[2] for l in llh])
            
    def elementFactor(self, u, v):
        antenna = self.antenna
        H = np.outer(np.interp(u, antenna.u, antenna.H.azimuth), 
                     np.interp(v, antenna.v, antenna.H.elevation))
        V = np.outer(np.interp(u, antenna.u, antenna.V.azimuth), 
                     np.interp(v, antenna.v, antenna.V.elevation)) 
        
        return H, V
    
    def arrayFactor(self, u, v, tTX, tRX):
        antenna = self.antenna
        wavenumber = 2*np.pi/c*antenna.fc
        UPos, VPos = np.meshgrid(antenna.azimuthPositions, 
                                 antenna.elevationPositions)
        MU = np.exp(-1j*wavenumber*np.einsum("ij,k -> kij", UPos, u))
        MV = np.exp(-1j*wavenumber*np.einsum("ij,k -> ijk", VPos, v))
        tPatt = np.einsum("ijk,jk,jkl -> il", MU, tTX, MV)
        rPatt = np.einsum("ijk,jk,jkl -> il", MU, tRX, MV)
        
        nmRX = np.sqrt(np.real(np.sum(tRX*tRX.conj())))
        
        return tPatt, rPatt/nmRX
    
    def update(self, xslxFile, modeName):
        antenna = self.antenna
        xlsx = self.xlsx
        xlsx.elementPattern = pd.read_excel(xslxFile, 
                                            sheet_name = "patternElement")
        
        """ Fix the following so that we haven't hard coded 13 """
        dd = pd.read_excel(xslxFile, 
                           sheet_name = "ExcitationCoeffs", 
                           header = [0,1],
                           na_filter = False,
                           index_col=[0,1,2])
        
        lindex = dd.index.tolist()
        lindex = [(l[0], 1, l[-1]) for l in lindex if l[1]==''] + [l for l in lindex if l[1] !='']
        repeated = list(set([z[0] for z in [[y for y in lindex if y == x] 
                                            for x in lindex] if len(z)>1]))
        for rp in repeated:
            lindex[lindex.index(rp)] = (rp[0], rp[1], None)
            
        dd.index = lindex
        
        subarray = pd.read_excel(xslxFile, 
                                 "ExcitCoeffError", 
                                 index_col = [0], 
                                 header = [0])
        
        sTX = np.array(eval(subarray.loc['TX coefficients error matrix size', 
                                         'Value']))
        
        sRX = np.array(eval(subarray.loc['RX coefficients error matrix size', 
                                         'Value']))
        
        aTX = np.array(eval(subarray.loc['Elements Taper H-pol abs', 
                                         'Value']))
        
        pTX = np.array(eval(subarray.loc['Elements Taper H-pol phase', 
                                         'Value']))
        
        aRX = np.array(eval(subarray.loc['Elements Taper V-pol abs', 
                                         'Value']))
        
        pRX = np.array(eval(subarray.loc['Elements Taper V-pol phase', 
                                         'Value']))
        
        wTX = 10**(aTX/10)*np.exp(1j*np.radians(pTX))
        wRX = 10**(aRX/10)*np.exp(1j*np.radians(pRX))
        
        swathNums = sorted(list(set([x[1] for x in dd.index])))
        
        mykeys = [x for x in dd.keys() if type(x[1])==int]
        swaths = []
        for swathNum in swathNums:
            swath = SimpleNamespace()
            qindex = [x for x in dd.index if x[1]==swathNum and x[-1] is not None]
            txGainKeys = [k for k in mykeys if k[0]=="Transmit gain"]
            txPhaseKeys = [k for k in mykeys if k[0]=="Transmit phase"]
            rxGainKeys = [k for k in mykeys if k[0]=="Receive gain"]
            rxPhaseKeys = [k for k in mykeys if k[0]=="Receive phase"]
            swath.swath = swathNum
            swath.TX = (dd.loc[qindex, txGainKeys].to_numpy()*
                  np.exp(1j*np.radians(dd.loc[qindex, txPhaseKeys].to_numpy())))
            swath.RX = (dd.loc[qindex, rxGainKeys].to_numpy()*
                  np.exp(1j*np.radians(dd.loc[qindex, rxPhaseKeys].to_numpy())))
            swath.TX = swath.TX.astype(np.complex128)
            swath.RX = swath.RX.astype(np.complex128)
            
            swath.rnSubArrayLenTX, swath.azSubArrayLenTX = (sTX/np.array(swath.TX.shape)).astype(int)
            swath.rnSubArrayLenRX, swath.azSubArrayLenRX = (sRX/np.array(swath.RX.shape)).astype(int)
            
            swath.TX = np.dot(swath.TX, spt.vectorM(wTX, swath.TX.shape[-1]))
            swath.RX = np.dot(swath.RX, spt.vectorM(wRX, swath.RX.shape[-1]))
                   
            swath.numSubArraysRn = int(swath.RX.shape[0]/swath.rnSubArrayLenRX)
            swath.numSubArraysAz = int(swath.RX.shape[1]/swath.azSubArrayLenRX)
            
            swaths.append(swath)
            
        self.swaths = swaths
            
        # for swath in swaths:
            
            #antenna.azSubArrayLen = wTX.shape[0]
            #antenna.rnSubArrayLen = 1
        # xlsx.coefficientsTX = pd.read_excel(xslxFile, 
        #                                     sheet_name = "ExcitationCoeffs", 
        #                                     skiprows = (swathNum-1)*13, 
        #                                     nrows=13)
        # xlsx.coefficientsRX = pd.read_excel(xslxFile, 
        #                                     sheet_name = "excitationsSA_RX",
        #                                     skiprows = (swathNum-1)*13, 
        #                                     nrows=13)
        """ End Fix """
        
        xlsx.swathTiming = pd.read_excel(xslxFile, sheet_name = modeName)
        xlsx.mode = pd.read_excel(xslxFile, 
                                  index_col = [0],
                                  sheet_name = modeName)
        xlsx.system = pd.read_excel(xslxFile,
                                    index_col = [0],
                                    sheet_name = "SystemParameters")
        # xlsx.antenna = pd.read_excel(xslxFile, sheet_name = "antenna")

        """ Get Element pattern """
        antenna.H = SimpleNamespace()
        antenna.V = SimpleNamespace()
        (antenna.u, 
         antenna.v, 
         antenna.H.azimuth, 
         antenna.H.elevation) = self.getElementPattern(xslxFile, 'H')
        (antenna.u, 
         antenna.v, 
         antenna.V.azimuth, 
         antenna.V.elevation) = self.getElementPattern(xslxFile, 'V')
        
        # dEpatternAbs = "elevation%sP_abs / 1" % pol
        # dEpatternPhs = "elevation%sP_phase / deg" % pol
        # dApatternAbs = "azimuth%sP_abs / 1" % pol
        # dApatternPhs = "azimuth%sP_phase / deg" % pol
        # dEAbs = xlsx.elementPattern[dEpatternAbs].to_numpy()
        # dEPhs = np.radians(xlsx.elementPattern[dEpatternPhs].to_numpy())
        # dAAbs = xlsx.elementPattern[dApatternAbs].to_numpy()
        # dAPhs = np.radians(xlsx.elementPattern[dApatternPhs].to_numpy())
        # antenna.u0 = xlsx.elementPattern["u"].to_numpy()
        # antenna.v0 = xlsx.elementPattern["v"].to_numpy()
        # antenna.H.ePattern = dEAbs*np.exp(1j*dEPhs)
        # antenna.aPattern = dAAbs*np.exp(1j*dAPhs)
        
        # antenna.azimuthPositions, _, antenna.azSubArrayLen, antenna.azimuthLength = self.getSpacing("azimuth")
        # antenna.elevationPositions, _, antenna.rnSubArrayLen, antenna.elevationLength  = self.getSpacing("elevation")
        (antenna.azimuthPositions, 
         antenna.elevationPositions) = self.getSpacing(nSe = sTX[0], 
                                                       nSa = sTX[-1])
        
        
        antenna.fc = xlsx.system.loc["Central frequency"]["Value"]*1e9
        return
    
    def _matchingKey(self, df, key):
        lkey = key.lower().replace(' ','')
        keys = [(x.lower().replace(' ',''), x) for x in df.keys()]
        match = [x[1] for x in keys if lkey in x[0]]
        if len(match) > 0:
            return match[0]
        else:
            return None
                 
    def getElementPattern(self, xslxFile, pol = 'H'):
        elementPattern = pd.read_excel(xslxFile, 
                                       sheet_name = "patternElement")
        dEpatternAbs = self._matchingKey(elementPattern, 
                                         "elevation%sP_abs / 1" % pol)
        dEpatternPhs = self._matchingKey(elementPattern, 
                                         "elevation%sP_phase / deg" % pol)
        dApatternAbs = self._matchingKey(elementPattern, 
                                         "azimuth%sP_abs / 1" % pol)
        dApatternPhs = self._matchingKey(elementPattern, 
                                         "azimuth%sP_phase / deg" % pol)
        try:
            u = elementPattern["u"].to_numpy()
            v = elementPattern["v"].to_numpy()
            dEAbs = elementPattern[dEpatternAbs].to_numpy()
            dEPhs = np.radians(elementPattern[dEpatternPhs].to_numpy())
            dAAbs = elementPattern[dApatternAbs].to_numpy()
            dAPhs = np.radians(elementPattern[dApatternPhs].to_numpy())
            return u, v, dAAbs*np.exp(1j*dAPhs), dEAbs*np.exp(1j*dEPhs)
        except KeyError:
            print("Pattern not in spreadsheet. Please check")
            return None, None, None, None
            
        
    # def elementFactor2D(self, u, v):
    #     antenna = self.instrument.antenna
    #     return np.outer(np.interp(u, antenna.u0, antenna.aPattern), 
    #                     np.interp(v, antenna.v0, antenna.ePattern))    
    
    # def arrayFactor2D(self, u, v, tTX, tRX):
    #     antenna = self.instrument.antenna
    #     wavenumber = 2*np.pi/c*antenna.fc
    #     UPos, VPos = np.meshgrid(antenna.azimuthPositions, 
    #                              antenna.elevationPositions)
    #     MU = np.exp(-1j*wavenumber*np.einsum("ij,k -> kij", UPos, u))
    #     MV = np.exp(-1j*wavenumber*np.einsum("ij,k -> ijk", VPos, v))
    #     tPatt = np.einsum("ijk,jk,jkl -> il", MU, tTX, MV)
    #     rPatt = np.einsum("ijk,jk,jkl -> il", MU, tRX, MV)
        
    #     nmRX = np.sqrt(np.real(np.sum(tRX*tRX.conj())))

    #     return tPatt, rPatt/nmRX


    
    # """ Some utility functions """
    # def getParameterFloat(self, sheet, paramName, column = "Value"):
    #     return float(sheet.loc[sheet[sheet.keys()[0]] == paramName][column])
    # def getParameterInt(self, sheet, paramName, column = "Value"):
    #     return int(sheet.loc[sheet[sheet.keys()[0]] == paramName][column])
    
    # def getSpacing(self, direction = "azimuth"):
    #     nSa = self.getParameterInt(self.xlsx.antenna, "Number of subarrays in %s" % direction)
    #     nRe = self.getParameterInt(self.xlsx.antenna,
    #                           "Number of radiating elements per subarray in %s" % direction)
    #     dim = "Length" if direction == "azimuth" else "Height"
    #     lE = self.getParameterFloat(self.xlsx.antenna, "%s of the antenna" % dim)
    #     nE = nSa*nRe
    #     return np.arange(nE)*lE/nE + lE/nE/2, nSa, nRe, lE
    
    def getSpacing(self, nSe = None, nSa = None):
        sys = self.xlsx.system
        nSa = nSa or sys.loc["Antenna columns (az)"]["Value"]
        nSe = nSe or sys.loc["Antenna rows (rg)"]["Value"]
        lA = sys.loc["Antenna size azimuth"]["Value"]
        lE = sys.loc["Antenna size range"]["Value"]
        pA = lA/nSa
        pE = lE/nSe
        
        spA = np.arange(nSa)*lA/nSa + pA/2
        spE = np.arange(nSe)*lE/nSe + pE/2
        return spA, spE
    
    """ Function to create expansion matrix """
    def identityM(self, indicator, k):
        m = len(indicator)
        iM = np.zeros((m, m*k), dtype = indicator.dtype)
        for i in range(k):
            iM[:, i::k] = np.diag(indicator)
        return iM
    
    def vectorM(self, vector, k):
        m = len(vector)
        iM = np.zeros((k, m*k), dtype = vector.dtype)
        for i in range(k):
            iM[i, (i*m):(i*m + m)] = vector
        return iM
    
    def readChannels(self):
        swaths = self.swaths
        antenna = self.antenna
        xlsx = self.xlsx
        mode = xlsx.mode
        
        for swath in swaths:
            k = swath.swath
            signal = SimpleNamespace()
            signal.pulse = SimpleNamespace()
            signal.swath = k
            signal.prf = mode.loc["PRF"][k]
            signal.pulse.bandwidth = mode.loc["Bandwidth"][k]*1e6
            signal.fs = mode.loc["Sampling frequency"][k]*1e6
            signal.pulse.length = mode.loc["Pulse Length"][k]/1e6
            signal.deltaR = 1.0/signal.fs*c/2
            signal.burstDuration = mode.loc["Burst duration"][k]
            signal.echoWindowLength = mode.loc["SWL"][k]/1e6
            signal.swst = mode.loc["SWST"][k]/1e6
            signal.nAsamples = int(signal.burstDuration*signal.prf)
            signal.nRsamples = int(signal.echoWindowLength*signal.fs)
            signal.nearRange = mode.loc["SWST"][k]/1e6
            signal.offNadir = mode.loc["Offnadir pointing"][k]
            
    
            # """ Get TX coefficients """
            # txAmpCols = ["SA Azimuth - %d" % (k+1) for k in range(5)]
            # txPhsCols = ["SA Azimuth - %d.1" % (k+1) for k in range(5)]
            # txAmp = xlsx.coefficientsTX[txAmpCols].to_numpy()
            # txPhs = np.radians(xlsx.coefficientsTX[txPhsCols].to_numpy())
            # TX = txAmp*np.exp(1j*txPhs)
    
            # """ Get RX coefficients """
            # rxAmpCols = ["SA Azimuth - %d" % (k+1) for k in range(5)]
            # rxPhsCols = ["SA Azimuth - %d.1" % (k+1) for k in range(5)]
            # rxAmp = xlsx.coefficientsRX[rxAmpCols].to_numpy()
            # rxPhs = np.radians(xlsx.coefficientsRX[rxPhsCols].to_numpy())
            # RX = rxAmp*np.exp(1j*rxPhs)
        
            # nElev = int(swath.RX.shape[0]/swath.rnSubArrayLenRX)
            # nAzim = int(swath.RX.shape[1]/swath.azSubArrayLenRX)
            iElev = np.array([self.identityM(channel, swath.rnSubArrayLenTX).T 
                              for channel in np.eye(swath.numSubArraysRn)])
            iAzim = np.array([self.identityM(channel, swath.azSubArrayLenTX)
                              for channel in np.eye(swath.numSubArraysAz)])

            tx_iElev = np.sum(iElev, axis=0)
            tx_iAzim = np.sum(iAzim, axis=(0,1))
            
            channels = []
            for mapsChan in np.arange(swath.numSubArraysAz):
                channel = SimpleNamespace()
                rx_iElev = np.sum(iElev, axis=0)
                rx_iAzim = np.sum(iAzim[mapsChan,:,:], axis=0)
        
                """ Expand the subarray weightings across all the elements """
                channel.RX = rx_iElev.dot(swath.RX)*np.tile(rx_iAzim, (rx_iElev.shape[0],1))
                channel.TX = tx_iElev.dot(swath.TX)*np.tile(tx_iAzim, (tx_iElev.shape[0],1))
                
                channel.delay = SimpleNamespace()
                channel.delay.sampling = 0
                wgtTx = np.abs(np.sum(channel.TX, axis=0))
                wgtTx/=np.sum(wgtTx)
                wgtRx = np.abs(np.sum(channel.RX, axis=0))
                wgtRx/=np.sum(wgtRx)
                channel.delay.baseline = (wgtTx + wgtRx).dot(antenna.azimuthPositions)/2
                channels.append(channel)
                
            signal.channels = channels
            swath.signal = signal

#%%
spt = sptRadar(roseLxsl, "DPN")

#%% Plot the antenna pattern
u = spt.antenna.u
v = [0]
RX = spt.swaths[0].signal.channels[0].RX
TX = spt.swaths[0].signal.channels[0].TX
aPtx, aPrx = spt.arrayFactor(u, v, TX, RX)
ePtH, ePtV = spt.elementFactor(u, v)

#%%
plt.figure()
twPtH = aPtx*aPrx*ePtH*ePtH
twPtV = aPtx*aPrx*ePtV*ePtV
#twPt = ePtH#*ePtH
plt.plot(u*180, 10*np.log10(np.abs(twPtH)))
plt.plot(u*180, 10*np.log10(np.abs(twPtV)))
plt.grid()
plt.show()

#%%
signal = spt.swaths[0].signal

#%%
def generateTimeDomainSignal(signal):
    rng = signal.nearRange*c/2 + signal.nRsamples/2*signal.deltaR
    ref = rad['acquisition']
    tmp = npmat.repmat(pointXYZ,ref['numAzimuthSamples'],1)
    rangeVectors = ref['satellitePositions'][1][:,0:3]-tmp
    rangeMagnitudes = np.sqrt(np.sum(rangeVectors*rangeVectors, axis=1))
    velocityVectors = ref['satellitePositions'][1][:,3:]
    velocityMagnitudes = np.sqrt(np.sum(velocityVectors*velocityVectors, axis=1))
    targetAngles = np.sum(rangeVectors*velocityVectors, axis=1)/rangeMagnitudes/velocityMagnitudes
    antennaWeight = twoWayArrayPattern(targetAngles, rad)
    fastTime = ref['nearRangeTime'] + np.arange(float(ref['numRangeSamples']))*ref['rangeSampleSpacing']
    allFastTimes = (npmat.repmat(fastTime,len(rangeMagnitudes),1).T
                    - 2.0/c*npmat.repmat(rangeMagnitudes,len(fastTime),1))
    chirp = chirpWaveform(rad['chirp']['pulseBandwidth'], rad['chirp']['length'])
    signal = chirp.sample(allFastTimes, rad['antenna']['fc'])*npmat.repmat(antennaWeight,len(fastTime),1)
    return signal
#%% get data from XML file
# rnSubArrayLen = 2
# azSubArrayLen = 12
# pq = etree.XMLParser(remove_blank_text=True)
# xml = etree.XML(base_ROSEL,parser=pq)
# fc = system["Value"][10]*1e9

# azimuthResolution = mode["Value"][3]
# rangeResolution = mode["Value"][4] #c/2.0/54.5624802e6 #vv.rn_resolution
# range_BANDWIDTH = c/2.0/rangeResolution
# deltaR = 1.0/1.2/range_BANDWIDTH #1.0/vv.rn_oversample/range_BANDWIDTH
# swathWidth = getParameterFloat(swathTiming, "Swath width", swath)*1e3 

# #vv.swath_width

# nE = antenna["Value"][3]*antenna["Value"][5]
# lE = antenna["Value"][9]
# dE = lE/nE
# elevationPositions = np.arange(nE)*lE/nE + lE/nE/2

# elevationPositions = np.array([float(f) for f in xml.find(".//elevationPositions").text.split()])
# azimuthPositions = np.array([float(f) for f in xml.find(".//azimuthPositions").text.split()])

#%% Write the element patterns
# antennaArray = xml.find(".//antennaArray")
# aElementPattern = etree.SubElement(antennaArray, "elementPattern")
# aElementPattern.set("axis", "azimuth")
# aElementPatternU = etree.SubElement(aElementPattern, "sinedirection")
# aElementPatternU.text = " ".join(["%0.3f" % x for x in u])
# aElementPatternAbs = etree.SubElement(aElementPattern, "magnitude")
# aElementPatternAbs.text = " ".join(["%0.3f" % x for x in u])

# #%% Compute the array factor
# def arrayFactor(cosangs, positions, TX, RX, k):
    
#     dM = np.exp(-1j*k*np.outer(cosangs, positions))
    
#     try:
#         tPattAF = dM.dot(tTX)
#         nmRX = np.diag(1/np.sqrt(np.real(np.sum(RX*RX.conj(), axis=0))))
#         rPattAF = dM.dot(tRX).dot(nmRX)
#     except ValueError:
#         tPattAF = dM.dot(tTX.T)
#         nmRX = np.diag(1/np.sqrt(np.real(np.sum(RX*RX.conj(), axis=1))))
#         rPattAF = dM.dot(tRX.T).dot(nmRX)
        
#     return tPattAF, rPattAF



# #%% Compute the array factor
# def arrayFactor2D(u, v, UPos, VPos, tTX, tRX, wavenumber):
#     MU = np.exp(-1j*wavenumber*np.einsum("ij,k -> kij", UPos, u))
#     MV = np.exp(-1j*wavenumber*np.einsum("ij,k -> ijk", VPos, v))
#     tPatt = np.einsum("ijk,jk,jkl -> il", MU, tTX, MV)
#     rPatt = np.einsum("ijk,jk,jkl -> il", MU, tRX, MV)
    
#     nmRX = np.sqrt(np.real(np.sum(RX*RX.conj())))

#     return tPatt, rPatt/nmRX

# def elementFactor2D(u,v):
#     return np.outer(np.interp(u, u0, aPattern), 
#                     np.interp(v, v0, ePattern))
    
# #%% function to create expansion matrix
# def identityM(indicator, k):
#     m = len(indicator)
#     iM = np.zeros((m,m*k))
#     for i in range(k):
#         iM[:, i::k] = np.diag(indicator)
#     return iM
        
# #%% Plot the azimuth element pattern
# plt.figure()
# plt.plot(u,20*np.log10(np.abs(aPattern)/np.max(np.abs(aPattern))))
# plt.ylim(-70,0)
# plt.grid()
# plt.xlabel('u')
# plt.ylabel('Gain (dB)')
# plt.show()

# #%%  
# wavenumber = 2*np.pi/c*fc

# UPos, VPos = np.meshgrid(azimuthPositions, elevationPositions)

# """ Define matrices to do the pre-summing of the subarrays """
# # rnSubArrayLen = 2
# # azSubArrayLen = 12
# iElev = np.array([identityM(channel, rnSubArrayLen).T 
#                   for channel in np.eye(TX.shape[0])])
# iAzim = np.array([identityM(channel, azSubArrayLen)
#                   for channel in np.eye(TX.shape[1])])

# my_iElev = np.sum(iElev, axis=0)
# my_iAzim = np.sum(iAzim, axis=0)

# #my_iElev = iElev[2,:,:]
# #my_iAzim = iAzim[2,:,:]

# """ Expand the subarray weightings across all the elements """
# tRX = my_iElev.dot(RX).dot(my_iAzim)
# tTX = my_iElev.dot(TX).dot(my_iAzim)

# #%% Compute the element pattern
# patternEF = np.outer(aPattern, ePattern)

# #%% Compute the one-way array factors
# tPattAF, rPattAF = arrayFactor2D(u, v, UPos, VPos, tTX, tRX, wavenumber)
# tPattAF, rPattAF = sptRadar.arrayFactor2D(u, v, dd.TX, dd.RX)

# #%% Plot the element factor pattern
# plt.figure()
# plt.imshow(20*np.log10(np.abs(patternEF)))
# plt.colorbar()
# plt.grid()
# plt.show()

# #%% Plot the array factor pattern
# plt.figure()
# plt.imshow(20*np.log10(np.abs(tPattAF)))
# plt.colorbar()
# plt.grid()
# plt.show()


# #%% Compute the composite patterns
# tPatt = patternEF*tPattAF
# rPatt = patternEF*rPattAF

# #%% Plot the two-way pattern
# plt.figure()
# plt.imshow(20*np.log10(np.abs(tPatt*rPatt)))
# plt.colorbar()
# plt.grid()
# plt.show()

# #%% Compute the directivity
# pTXav = np.real(np.mean(tPatt*tPatt.conj(), axis=0))
# tDirectivity = tPatt.dot(np.diag(1/np.sqrt(pTXav)))
# rDirectivity = v.size*rPatt.dot(np.diag(1/np.linalg.norm(rPatt, axis=0)))

# #%% Compute the one way patterns in azimuth
# tPwr = np.real(np.sum(TX*TX.conj(), axis=0))
# rNrm = np.sqrt(np.real(np.sum(RX*RX.conj(), axis=0)))
# nmTX = np.diag(1/np.linalg.norm(np.abs(TX), axis=0))
# nmRX = np.diag(1/rNrm)
# #tPatt = (np.tile(ePattern, (TX.shape[1], 1)).T)*(dM.dot(tTX))#.dot(nmTX)
# pTXav = np.real(np.mean(tPatt*tPatt.conj(), axis=0))
# tDirectivity = tPatt.dot(np.diag(1/np.sqrt(pTXav)))
# #rPatt = (np.tile(ePattern, (RX.shape[1], 1)).T)*(dM.dot(tRX)).dot(nmRX)
# rDirectivity = v.size*rPatt.dot(np.diag(1/np.linalg.norm(rPatt, axis=0)))

# #%% Plot TX directivity
# plt.figure()
# plt.plot(v, 20*np.log10(np.abs(tPatt)))
# plt.grid()
# plt.xlabel("v")
# plt.ylabel("Directivity (dB)")
# plt.show()

# plt.figure()
# plt.plot(v, 20*np.log10(np.abs(tDirectivity)))
# plt.grid()
# plt.xlabel("u")
# plt.ylabel("Directivity (dB)")
# plt.show()

# #%%
# eBeamWeights = np.ones((24,))
# eBeamWeights /= np.linalg.norm(eBeamWeights)

# #%%
# def readSignalDataElement(eBeamWeights, TX, RX, channel = 1):
#     iElev = np.array([identityM(channel, rnSubArrayLen).T 
#                       for channel in np.eye(TX.shape[0])])
#     iAzim = np.array([identityM(channel, azSubArrayLen)
#                       for channel in np.eye(TX.shape[1])])
#     my_iElev = np.sum(iElev, axis=0)
#     my_iAzim = np.sum(iAzim, axis=0)
#     tTX = my_iElev.dot(TX).dot(my_iAzim)
#     txCols = eBeamWeights.dot(tTX)
#     refRxCols = eBeamWeights.dot(my_iElev.dot(RX).dot(iAzim[0,:,:]))
#     rxCols = eBeamWeights.dot(my_iElev.dot(RX).dot(iAzim[channel,:,:]))
#     # txCols = np.array([m*np.complex(np.cos(p), np.sin(p)) for 
#     #           m,p in zip(sconv.toPowerArray('.//magnitude', tx), 
#     #                      sconv.toAngleArray('.//phase', tx))])
#     # txuZero = sconv.toAngle('.//u0', tx)
#     # rx = sconv.getXMLElement('.//receiveConfiguration', xmlroot)
#     # rxCols = np.array([m*np.complex(np.cos(p), np.sin(p)) for 
#     #           m,p in zip(sconv.toPowerArray('.//magnitude', rx), 
#     #                      sconv.toAngleArray('.//phase', rx))])
#     # rxuZero = sconv.toAngle('.//u0', rx)
    
#     # txMag = np.array(sconv.toPowerArray('.//magnitude', tx))
#     # txDelay = np.array(sconv.toDurationArray('.//truedelay', tx))
#     # rxMag = np.array(sconv.toPowerArray('.//magnitude', rx))
#     # rxDelay = np.array(sconv.toDurationArray('.//truedelay', rx))
#     # rdrfilename = sconv.toString('.//filename', xmlroot)
#     prf = getParameterFloat(swathTiming, "PRF", swath)
#     pulseDuration = getParameterFloat(swathTiming, "Pulse Length", swath)*1e-6
#     pulseBandwidth = getParameterFloat(swathTiming, "Bandwith in Range", swath)*1e6
#     oversampleRn = getParameterFloat(mode, "Coefficient Over Sampling - Range")
#     fs = pulseBandwidth*oversampleRn
#     burstDuration = getParameterFloat(swathTiming, "Duration of the burst", swath)
#     nearRange = getParameterFloat(swathTiming, "Minimum slant range", swath)*1e3
#     radar = {'acquisition': {'startTime': np.datetime64("2022-01-01T00:00:00.000000000"),
#                              'numAzimuthSamples': int(prf*burstDuration+0.5),
#                              'numRangeSamples': int(fs*pulseDuration + 0.5),
#                              'prf': prf,
#                              'nearRangeTime': nearRange*2/c,
#                              'rangeSampleSpacing': 1.0/fs*c/2},
#              'chirp': {'length': pulseDuration,
#                        'pulseBandwidth': pulseBandwidth},
#              'mode': {'txColumns': txCols, 
#                       'rxColumns': rxCols,
#                       'txMagnitude': np.abs(txCols),
#                       'txDelay': txDelay,
#                       'txuZero': txuZero,
#                       'rxuZero': rxuZero,
#                       'rxMagnitude': np.abs(rxCols),
#                       'rxDelay': rxDelay},
#              'delay': {'baseline': (sum(np.abs(txCols)*antenna['azimuthPositions'])/sum(abs(txCols))/2.0 
#                           + sum(np.abs(rxCols)*antenna['azimuthPositions'])/sum(abs(rxCols))/2.0)},
#              'filename': rdrfilename
#              }
#     return radar

# #%% Write to file
# def createXMLStructure(folder): 
#     for fd in ["simulation_data", 
#                "simulation_plots",
#                os.path.join("simulation_data", "raw"),
#                os.path.join("simulation_data", "mchan_processed"),
#                os.path.join("simulation_data", "wk_processed")]:
#         tfd = os.path.join(folder, fd)
#         if not os.path.exists(tfd):
#             try:
#                 os.makedirs(tfd)
#             except IOError as ie:
#                 print(ie)
#                 return False
#     return True

# #%% Write to file
# def writeToXML(xml_file_name, xml):  
#     with open(xml_file_name,'w') as outfile:
#         outfile.write(etree.tostring(xml, pretty_print=True).decode())
    
# #%% Generate the XML object. This code needs refactoring and documentation
# def generateXML(vv,
#                 va = 7500.0,
#                 nearRange = 5.746681592636996e-03,
#                 element_length = 0.04):
    
#     print("SURE configuration generator")
#     numAziSamples = 8192
#     numRngSamples = 1024
    
#     #%% Generate the base XML object
#     pq = etree.XMLParser(remove_blank_text=True)
#     xml = etree.XML(base_ROSEL,parser=pq)
#     fc = float(xml.find(".//carrierFrequency").text)
        
#     #%% Set some defaults
#     # azimuthResolution = 11.8 #vv.az_resolution
#     # rangeResolution = c/2.0/54.5624802e6 #vv.rn_resolution
#     # range_BANDWIDTH = c/2.0/rangeResolution
#     # deltaR = 1.0/1.2/range_BANDWIDTH #1.0/vv.rn_oversample/range_BANDWIDTH
#     M = 4
        
#     number_channels = M + 1
#     # print("Number of channels: (M+1) = %d" % number_channels)
#     # print("Ideal PRF: %0.4f Hz" % (va/(M+1)/azimuthResolution))
    
#     #%% Compute some collection parameters
#     number_beams = 1 #number_channels
#     range_BANDWIDTH = swathTiming[swath][12]*1e6
#     pulse_duration = swathTiming[swath][17]*1e-6
#     fs = mode["Value"][6]*range_BANDWIDTH
#     deltaR = 1.0/fs
#     vg = swathTiming[swath][11]
#     target_DOPPLER = swathTiming[swath][18]
#     azimuthResolution = vg/target_DOPPLER
#     aziDoppler = va/azimuthResolution
#     beam_DOPPLER = aziDoppler #/number_channels
#     prf = swathTiming[swath][13]
#     near_RANGE = swathTiming[swath][6]*1e3
#     far_RANGE = swathTiming[swath][7]*1e3
#     channel_LEN = np.round(swathTiming[swath][10]*1e3/vg*prf)
#     # channel_LEN = int(np.round(far_RANGE*(c/fc)
#     #                            *(prf/number_channels)
#     #                            *(beam_DOPPLER)
#     #                            /2.0/(va**2)))
#     # print("Doppler bandwidth per beam: %f Hz" % beam_DOPPLER)
#     print("Number of samples/burst: %d" % channel_LEN)
#     # print("Number of samples/aperture (channels and beams combined): %d" 
#     #       % (channel_LEN*number_channels*number_beams))
#     print("Near range: %f m" % near_RANGE)
#     print("Far range: %f m" % far_RANGE)
#     print("Number of channels: %d" % number_channels)
#     print("Number of beams: %d" % number_beams)
#     print("Slow time PRF: %f Hz" % prf)
    
    
#     #%% Compute the azimuth positions
#     # numAziSamples = 2*channel_LEN*(2+number_channels*number_beams)
    
#     # subarray_length = number_channels*azimuthResolution*2.0
    
    
#     # # Compute the channel tables for each pulse in the g vector. 
#     # # i.e. put in the squint
#     # subarray_elements = int(subarray_length/element_length)
#     # element_spacing = subarray_length/subarray_elements
#     # print("Beam spread interval: %f Hz" % beam_DOPPLER)
#     # print("Antenna Element Length: %f m" % element_length)
#     # print("Subarray Length: %f m" % subarray_length)
#     # subarray_centres = [(pos-(number_channels-1)/2.0)*subarray_length 
#     #                     for pos in range(number_channels)]
#     # azimuthPositions = [[subpos + element_spacing/2.0 - 
#     #                   subarray_length/2.0 + k*element_spacing 
#     #                   for k in range(subarray_elements)] 
#     #                  for subpos in subarray_centres]
    
#     #%% Write the azimuth positions, lengths and powers
    
#     # flattened_azimuthPositions = reduce(lambda x,y: x+y, azimuthPositions)
#     # str_azi_pos = ["%0.6f" % pos for pos in flattened_azimuthPositions]
#     # str_azi_len = ["%0.6f" % element_length 
#     #                for pos in flattened_azimuthPositions]
#     # str_azi_pwr = ["%0.1f" % vv.element_power 
#     #                for pos in flattened_azimuthPositions]
#     # print_azi=False
#     # if print_azi:
#     #     print("Azimuth positions:")
#     #     print("----------------------------------------------------")
#     #     print(" ".join(str_azi_pos))
#     # azimuthPosNode = xml.find('.//azimuthPositions')
#     # azimuthPosNode.text = " ".join(str_azi_pos)
#     # azimuthLenNode = xml.find('.//azimuthElementLengths')
#     # azimuthLenNode.text = " ".join(str_azi_len)
#     # azimuthPwrNode = xml.find('.//transmitPowers')
#     # azimuthPwrNode.text = " ".join(str_azi_pwr)
    
#     #%% Define some parameters from the spreadsheet
#     beam_DC = [0]
    
#     #%% Calculate the Doppler centroids for each beam
#     beam_DC = [beam_DOPPLER*(pos-(number_beams-1)/2.0) 
#                for pos in range(number_beams)]
#     reducedPRF = prf/len(beam_DC)
#     print("Doppler frequencies:")
#     print("----------------------------------------------------")
#     print(beam_DC)
#     txChan = 1 %int((number_channels-1)/2)
     
#     #%% Generate the reference time from which to compute the offsets. e.t.c.
#     reference_TIME = datetime.combine(date(2015,1,1),time(0,0,0))
#     sample_TIMES = [reference_TIME + timedelta(seconds=float(s)/prf) 
#                     for s in range(len(beam_DC))]
#     nref_TIME = np.datetime64("2015-01-01T00:00:00")
#     nsample_TIMES = [nref_TIME
#                      + np.timedelta64(int(np.round(1.0e9*float(s)/prf)), 'ns') 
#                      for s in range(len(beam_DC))]
    
#     #%% Calculate how many extra range samples we'll need
#     u = c/fc/(4*azimuthResolution)
#     sim_range_samples = 2*(far_RANGE - near_RANGE)/c/deltaR
#     RMC_range = (far_RANGE/c*2)*(np.sqrt(1.0+u**2)-1)/deltaR + numRngSamples
    
#     min_range_samples = int(RMC_range)
#     # Add the offset due to range migration to the nim range samples
#     print("RCM additional range: %f" % (RMC_range))
#     print("Full simulation required range samples: %f" % sim_range_samples)
#     print("New computation: %f" 
#           % (2.0*far_RANGE/c*(1.0/np.cos(c/fc/azimuthResolution) - 1)/deltaR))
#     print("Minumum number of range samples in a pulse: %f" % min_range_samples)
#     liked_range_samples = np.array(sorted(
#         np.outer(3**(np.arange(5)), 2**(9 + np.arange(10))).flatten()))
#     numRngSamples = liked_range_samples[liked_range_samples>min_range_samples][0]
#     print("Number of range samples for this simulation: %d" % numRngSamples)
#     print("Number of azimuth samples per signal file: %d" 
#           % int(numAziSamples/number_beams))
    
#     #%% Estimate the size of the data
#     numSigElements = number_channels*len(beam_DC)
#     fileSize = 16*numRngSamples*numAziSamples/number_beams
#     print("Total number of signal elements: %d" % numSigElements)
#     print("Size of each signal data file: %d" % int(fileSize))
#     print("Size of all signal files in memory: %f (GB)" 
#           % float(fileSize*numSigElements/1e9))
    
#     predicted_bands = np.arange(-int(numSigElements/2)-2,
#                                 int(numSigElements/2)+1+2)
#     print("Predicted number of bands: %d" % len(predicted_bands))
#     mchanFileSize = len(predicted_bands)*fileSize
#     print("Predicted size of multi-channel processed file: %f (GB)" 
#           % float(mchanFileSize/1e9))
#     print("Peak memory requirement (approx): %f (GB)" 
#           % float((mchanFileSize+fileSize*numSigElements)/1e9))
    
#     #%% Create the mode XML mode mode snippet
#     for fd,n,thisTime,isoTime in zip(beam_DC, 
#                                      range(len(beam_DC)), 
#                                      sample_TIMES, 
#                                      nsample_TIMES):
#         phases = [pi*x*fd/va for x in azimuthPositions]
#         truedelay = [1.0e9*x/(2.0*va)*fd/fc for x in azimuthPositions]
                
#         for channel in range(number_channels):
#             signalData = etree.Element("signalData")
#             beam_Angle = -c*fd/(2.0*va*fc)
            
#             # Add some nodes
#             timeNode = etree.SubElement(signalData,"azimuthStartTime")
#             timeNode.set("iso", str(isoTime))
#             yearNode = etree.SubElement(timeNode,"year")
#             monthNode = etree.SubElement(timeNode,"month")
#             dayNode = etree.SubElement(timeNode,"day")
#             hourNode = etree.SubElement(timeNode,"hour")
#             minuteNode = etree.SubElement(timeNode,"minute")
#             secondNode = etree.SubElement(timeNode,"sec")
#             usecNode = etree.SubElement(timeNode,"usec")
#             systemNode = etree.SubElement(timeNode,"time_system")
    
#             yearNode.text = '%0.4d'%thisTime.year
#             monthNode.text = '%0.2d'%thisTime.month
#             dayNode.text = '%0.2d'%thisTime.day
#             hourNode.text = '%0.2d'%thisTime.hour
#             minuteNode.text = '%0.2d'%thisTime.minute
#             secondNode.text = '%0.2d'%thisTime.second
#             usecNode.text = '%0.6d'%thisTime.microsecond
#             systemNode.text = '2'
    
#             numRangeSamples = etree.SubElement(signalData,"numRangeSamples")
#             numRangeSamples.set("unit","Pixels")
#             numRangeSamples.text = "%d" % numRngSamples
    
#             numAzimuthSamples = etree.SubElement(signalData,
#                                                  "numAzimuthSamples")
#             numAzimuthSamples.set("unit","Pixels")
#             numAzimuthSamples.text = "%d" % (numAziSamples/number_beams)
    
    
#             nearRangeTime = etree.SubElement(signalData,"nearRangeTime")
#             nearRangeTime.set("unit","s")
#             nearRangeTime.text = "%0.10e" % nearRange
    
#             rangeSampleSpacing = etree.SubElement(signalData,
#                                                   "rangeSampleSpacing")
#             rangeSampleSpacing.set("unit","s")
#             rangeSampleSpacing.text = "%0.6e" % deltaR
    
#             pulseRepetitionFrequency = etree.SubElement(signalData,
#                                                         "pulseRepetitionFrequency")
#             pulseRepetitionFrequency.set("unit","Hz")
#             pulseRepetitionFrequency.text = "%0.4f" % reducedPRF
    
#             pulseBandwidth = etree.SubElement(signalData,"pulseBandwidth")
#             pulseBandwidth.set("unit","MHz")
#             pulseBandwidth.text = "%f" % (range_BANDWIDTH/1.0e6)
    
#             pulseDuration = etree.SubElement(signalData,"pulseDuration")
#             pulseDuration.set("unit","us")
#             pulseDuration.text = "%0.4f" % pulse_duration
            
#             # Add the radar configuration
#             config = etree.Element("radarConfiguration")
#             config.set("channel", "channel%d-%d" % (channel,n))
#             Tx = etree.Element("transmitConfiguration")
#             Rx = etree.Element("receiveConfiguration")
#             Tpol = etree.Element("polarization")
#             Tpol.text = "H"
#             Rpol = etree.Element("polarization")
#             Rpol.text = "H"
#             txChanMask = [int(bool(x==txChan)) for x in range(number_beams)]
#             txWeight = [[y for x in azimuthPositions] for y in txChanMask]
#             txMag = etree.Element("magnitude")
#             txMag.set("unit", "natural")
#             txMag.text = " ".join(["%d" % x for x in 
#                                    reduce(lambda x,y: x+y, txWeight)])
#             txPhase = etree.Element("phase")
#             txPhase.set("unit", "radians")
#             txPhase.text = " ".join(["%0.3f" % x for x in 
#                                      reduce(lambda x,y: x+y, phases)])
#             txDelay = etree.Element("truedelay")
#             txDelay.set("unit", "ns")
#             txDelay.text = " ".join(["%0.6f" % x for x in 
#                                      reduce(lambda x,y: x+y, truedelay)])
#             rxChanMask = [int(bool(x==channel)) for x in range(number_beams)]
#             rxWeight = [[y for x in azimuthPositions[channel]] 
#                         for y in rxChanMask]
#             rxMag = etree.Element("magnitude")
#             rxMag.set("unit", "natural")
#             rxMag.text = " ".join(["%d" % x for x in 
#                                    reduce(lambda x,y: x+y, rxWeight)])
#             rxPhase = etree.Element("phase")
#             rxPhase.set("unit", "radians")
#             rxPhase.text = " ".join(["%0.3f" % x for x in 
#                                      reduce(lambda x,y: x+y, phases)])
#             rxDelay = etree.Element("truedelay")
#             rxDelay.set("unit", "ns")
#             rxDelay.text = " ".join(["%0.6f" % x for x in 
#                                      reduce(lambda x,y: x+y, truedelay)])
#             txuZero = etree.Element("u0")
#             txuZero.set("unit", "radians")
#             txuZero.text = "%0.6e" % beam_Angle
#             rxuZero = etree.Element("u0")
#             rxuZero.set("unit", "radians")
#             rxuZero.text = "%0.6e" % beam_Angle
#             Tx.append(Tpol)
#             Tx.append(txuZero)
#             Tx.append(txMag)
#             Tx.append(txPhase)
#             Tx.append(txDelay)
#             Rx.append(Rpol)
#             Rx.append(rxuZero)
#             Rx.append(rxMag)
#             Rx.append(rxPhase)
#             Rx.append(rxDelay)
#             config.append(Tx)
#             config.append(Rx)
#             signalData.append(config)
#             thisfilename = etree.SubElement(signalData,"filename")
#             thisfilename_text = "%s_%s_c%db%d.npy" % (os.path.split(vv.config_xml)[-1].split(".")[0], 
#                                                       vv.file_data_domain, 
#                                                       channel, 
#                                                       n)
#             thisfilename.text = os.path.sep.join([os.path.split(vv.config_xml)[0],
#                                                   "simulation_data",
#                                                   "raw",
#                                                   thisfilename_text])
#             xml.append(signalData)
            
#     return xml

# #%% Modify configurations
# def modifyRadarConfigurations(gmtiDataPool, radarConfigURL, directory):
#     # Open the radar configuration file and parse
#     radarConfig = etree.parse(urllib.urlopen(radarConfigURL))
#     radarConfigurations = radarConfig.xpath('//radarConfiguration')

#     # Define the default pulse type
#     pulse = {'pulseId': "-", 'pulseString': "Unknown", 'pulseDuration': "0", 'pulseBwd': "0"}

#     # Get all the signalData elements
#     currentConfigs = gmtiDataPool.xpath('//signalData')

#     print("Current configurations are: ")
#     printSignalRadarConfigurations(currentConfigs)
#     currentConfig = int(raw_input("Enter the number of the configuration you wish to replace/add (0 for none)? "))
#     while((currentConfig > 0) and (currentConfig <= len(currentConfigs))):
#         targetConfig = int(raw_input("Enter the number of the desired radar configuration (-1 for a list)?"))
#         while((targetConfig < 1) or (targetConfig > len(radarConfigurations))):
#             printRadarConfigurations(radarConfigurations)
#             targetConfig = int(raw_input("Enter the number of the configuration you wish to replace it with (-1 for a list)? "))

#         # Rename the element so we dont need to write so much
#         configParent = currentConfigs[currentConfig-1]

#         # Find and remove the current configuration if it is present
#         toRemove = configParent.xpath('radarConfiguration')
#         if(len(toRemove)>0):
#             # record the current pulse type
#             pulse = {'pulseId': configParent.xpath('radarConfiguration/pulse/pulseId')[0].text,
#                      'pulseString': configParent.xpath('radarConfiguration/pulse/pulseString')[0].text,
#                      'pulseDuration': configParent.xpath('radarConfiguration/pulse/pulseDuration')[0].text,
#                      'pulseBwd': configParent.xpath('radarConfiguration/pulse/pulseBwd')[0].text}

#             # Remove the old element
#             configParent.remove(toRemove[0])
#         else:
#             dataFile = configParent.xpath('dataFile')
#             if(len(dataFile)>0):
#                 pulse = getPulseType(os.path.join(directory,dataFile[0].text).replace('SarImage_0.tif','PARM_IngestParameters.self'))

#         # Add the new config. First locate the point where it will go
#         # The schema says there should be a dopplerCentroid element
#         dopEl = configParent.find('dopplerCentroid')
#         dopEl.addprevious(deepcopy(radarConfigurations[targetConfig-1]))

#         # Now add the pulse element
#         rConf = configParent.find('radarConfiguration')
#         pulseElement = etree.Element("pulse")
#         pulseId = etree.SubElement(pulseElement, "pulseId")
#         pulseSt = etree.SubElement(pulseElement, "pulseString")
#         pulseDu = etree.SubElement(pulseElement, "pulseDuration")
#         pulseBw = etree.SubElement(pulseElement, "pulseBwd")
#         pulseId.text = pulse['pulseId']
#         pulseSt.text = pulse['pulseString']
#         pulseDu.text = pulse['pulseDuration']
#         pulseBw.text = pulse['pulseBwd']
#         rConf.insert(0, pulseElement)

#         #currentConfigs[currentConfig-1].getparent().remove(currentConfigs[currentConfig-1])
#         currentConfigs = gmtiDataPool.xpath('//signalData')

#         # Ask for the next config to change
#         print("Current configurations are: ")
#         printSignalRadarConfigurations(currentConfigs)
#         currentConfig = int(raw_input("Enter the number of the configuration you wish to change (0 for none)? "))

#%% Print configurations
# def printRadarConfigurations(radarConfigurations):
#     for config, index in zip(radarConfigurations, range(len(radarConfigurations))):
#         tag = config.get('channel')
#         txPol = config.xpath('transmitConfiguration/polarization')
#         txCols = config.xpath('transmitConfiguration/magnitude')
#         rxPol = config.xpath('receiveConfiguration/polarization')
#         rxCols = config.xpath('receiveConfiguration/magnitude')
#         print("%0.2d"%(index+1) + ': ' + ' --> ' + txPol[0].text + ': ' + txCols[0].text + ' <-- ' + rxPol[0].text + ': '+ rxCols[0].text + ' ' + tag)

