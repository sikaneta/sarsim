#!/usr/bin/env python2.7

#%%
""" This python program should generate the config files for burst mode multi 
    or single channel data.
    The program generates an xml file to be processed in MATLAB """

import numpy as np
from datetime import datetime, date, time, timedelta
from math import pi
import urllib
from copy import deepcopy
import lxml.etree as etree
from functools import reduce
import os
from scipy.constants import c
from generateXML.base_XML import base_SentinelNG
import pandas as pd
import os
import matplotlib.pyplot as plt

#%% Coefficients file
SentinelNGpath = r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\Sentinel-NG\ISRR\SW01_aresys_Instrument Performance Mathematical Models including software\S1NG_pfsoftware\Inputs"
SentinelNGxsl = os.path.join(SentinelNGpath, "S1NG_CP IW Ish.xlsx")
elementPattern = pd.read_excel(SentinelNGxsl, sheet_name = "Antenna1D")

#%% Excitation coefficients
swath = 2
elementPattern = pd.read_excel(SentinelNGxsl, sheet_name = "Antenna1D")

#%% TX Abs
txAmp = pd.read_excel(SentinelNGxsl, 
                      sheet_name = "ExcitationCoeffs",
                      skiprows = 1 + (swath-1)*25,
                      usecols = "D:L").to_numpy()

#%% TX Phase
txPhs = pd.read_excel(SentinelNGxsl, 
                      sheet_name = "ExcitationCoeffs",
                      skiprows = 1 + (swath-1)*25,
                      usecols = "N:V").to_numpy()
txPhs = np.radians(txPhs)

#%% RX Abs
rxAmp = pd.read_excel(SentinelNGxsl, 
                      sheet_name = "ExcitationCoeffs",
                      skiprows = 1 + (swath-1)*25,
                      usecols = "X:AF").to_numpy()

#%% RX Phase
rxPhs = pd.read_excel(SentinelNGxsl, 
                      sheet_name = "ExcitationCoeffs",
                      skiprows = 1 + (swath-1)*25,
                      usecols = "AH:AP").to_numpy()
rxPhs = np.radians(rxPhs)

# #%% Get TX coefficients
# txAmpCols = ["SA Azimuth - %d" % (k+1) for k in range(5)]
# txPhsCols = ["SA Azimuth - %d.1" % (k+1) for k in range(5)]

# txAmp = coefficientsTX[txAmpCols].to_numpy()
# txPhs = np.radians(coefficientsTX[txPhsCols].to_numpy())

TX = txAmp*np.exp(1j*txPhs)

#%% Get RX coefficients
# rxAmpCols = ["SA Azimuth - %d" % (k+1) for k in range(5)]
# rxPhsCols = ["SA Azimuth - %d.1" % (k+1) for k in range(5)]

# rxAmp = coefficientsRX[rxAmpCols].to_numpy()
# rxPhs = np.radians(coefficientsRX[rxPhsCols].to_numpy())

RX = rxAmp*np.exp(1j*rxPhs)

#%% Get Element pattern
dEpatternAbs = "ElmeanAmplitudeH"
dApatternAbs = "AZmeanAmplitudeH"
dEAbs = elementPattern[dEpatternAbs].to_numpy()[1:].astype('double')
dAAbs = elementPattern[dApatternAbs].to_numpy()[1:].astype('double')
u = elementPattern["u"].to_numpy()[1:].astype('double')
v = elementPattern["v"].to_numpy()[1:].astype('double')
ePattern = dEAbs#*np.exp(1j*dEPhs)
aPattern = dAAbs#*np.exp(1j*dAPhs)

#%% Plot the azimuth element pattern
plt.figure()
plt.plot(u,20*np.log10(np.abs(aPattern)/np.max(np.abs(aPattern))))
plt.ylim(-70,0)
plt.grid()
plt.xlabel('u')
plt.ylabel('Gain (dB)')
plt.title('Element Pattern (Azimuth) Sentinel-1NG')
plt.show()

#%% get data from XML file
pq = etree.XMLParser(remove_blank_text=True)
xml = etree.XML(base_SentinelNG,parser=pq)
fc = float(xml.find(".//carrierFrequency").text)

# azimuthResolution = 11.8 #vv.az_resolution
# rangeResolution = c/2.0/54.5624802e6 #vv.rn_resolution
# range_BANDWIDTH = c/2.0/rangeResolution
# deltaR = 1.0/1.2/range_BANDWIDTH #1.0/vv.rn_oversample/range_BANDWIDTH
# swathWidth = 88e3 #vv.swath_width

elevationPositions = np.array([float(f) for f in xml.find(".//elevationPositions").text.split()])
azimuthPositions = np.array([float(f) for f in xml.find(".//azimuthPositions").text.split()])

#%% Write the element patterns
# antennaArray = xml.find(".//antennaArray")
# aElementPattern = etree.SubElement(antennaArray, "elementPattern")
# aElementPattern.set("axis", "azimuth")
# aElementPatternU = etree.SubElement(aElementPattern, "sinedirection")
# aElementPatternU.text = " ".join(["%0.3f" % x for x in u])
# aElementPatternAbs = etree.SubElement(aElementPattern, "magnitude")
# aElementPatternAbs.text = " ".join(["%0.3f" % x for x in u])

#%% Compute the array factor
def arrayFactor(cosangs, positions, TX, RX, k):
    
    dM = np.exp(-1j*k*np.outer(cosangs, positions))
    
    try:
        tPattAF = dM.dot(tTX)
        nmRX = np.diag(1/np.sqrt(np.real(np.sum(RX*RX.conj(), axis=0))))
        rPattAF = dM.dot(tRX).dot(nmRX)
    except ValueError:
        tPattAF = dM.dot(tTX.T)
        nmRX = np.diag(1/np.sqrt(np.real(np.sum(RX*RX.conj(), axis=1))))
        rPattAF = dM.dot(tRX.T).dot(nmRX)
        
    return tPattAF, rPattAF



#%% Compute the array factor
def arrayFactor2D(u, v, UPos, VPos, tTX, tRX, wavenumber):
    MU = np.exp(-1j*wavenumber*np.einsum("ij,k -> kij", UPos, u))
    MV = np.exp(-1j*wavenumber*np.einsum("ij,k -> ijk", VPos, v))
    tPatt = np.einsum("ijk,jk,jkl -> il", MU, tTX, MV)
    rPatt = np.einsum("ijk,jk,jkl -> il", MU, tRX, MV)
    
    nmRX = np.sqrt(np.real(np.sum(RX*RX.conj())))

    return tPatt, rPatt/nmRX

#%% function to create expansion matrix
def identityM(indicator, k):
    m = len(indicator)
    iM = np.zeros((m,m*k))
    for i in range(k):
        iM[:, i::k] = np.diag(indicator)
    return iM

#%%  
wavenumber = 2*np.pi/c*fc

UPos, VPos = np.meshgrid(azimuthPositions, elevationPositions)

""" Define matrices to do the pre-summing of the subarrays """
rnSubArrayLen = 6
azSubArrayLen = 1
iElev = np.array([identityM(channel, rnSubArrayLen) 
                  for channel in np.eye(int(TX.shape[0]/rnSubArrayLen))])
iAzim = np.array([identityM(channel, azSubArrayLen).T
                  for channel in np.eye(TX.shape[1])])

my_iElev = np.sum(iElev, axis=0)
my_iAzim = np.sum(iAzim, axis=0)

# my_iElev = iElev[2,:,:]
# my_iAzim = iAzim[2,:,:]
""" Expand the subarray weightings across all the elements """
el_channel = 0
az_channel = 0
tRX = np.outer(my_iElev[el_channel,:], my_iAzim[:,az_channel])
tTX = TX

#%% Compute the element pattern
patternEF = np.outer(aPattern, ePattern)

#%% Compute the one-way array factors
tPattAF, rPattAF = arrayFactor2D(u, v, UPos, VPos, TX, tRX, wavenumber)

#%% Plot the element factor pattern
plt.figure()
plt.imshow(20*np.log10(np.abs(patternEF)))
plt.colorbar()
plt.grid()
plt.show()

#%% Plot the array factor pattern
plt.figure()
plt.imshow(20*np.log10(np.abs(tPattAF)))
plt.colorbar()
plt.grid()
plt.show()


#%% Compute the composite patterns
tPatt = patternEF*tPattAF
rPatt = patternEF*rPattAF
twPatternAbs = 20*np.log10(np.abs(tPatt*rPatt))
twPatternAbs -= np.max(twPatternAbs)

#%% Plot cuts along patterns
plt.figure()
plt.plot(u, twPatternAbs[:,500])
plt.grid()
plt.title('Azimuth pattern')
plt.xlabel('u')
plt.ylabel('Response (dB)')
plt.show()

#%%
satvel = 7590
wavelength = c/fc
prf = 1501
fd = 2*u*satvel/wavelength
y = twPatternAbs[:,500]
plt.figure()
plt.plot(2*u*satvel/wavelength, y)
plt.grid()
plt.title('2-way azimuth pattern IW swath 2')
plt.xlabel('f_d (Hz)')
plt.ylabel('Response (dB)')
idx = np.argwhere(np.abs(fd) <= 9*prf/2)[:,0]
fdup = np.arange(fd[idx[0]], fd[idx[-1]], 1)
yup = np.interp(fdup, fd[idx], y[idx])
colors = ['red', 'indianred', 'darksalmon', 'sienna', 'coral']
for k in range(-4,5):
    idx = np.argwhere(np.abs(fdup-k*prf) < prf/2)[:,0]
    plt.fill_between(fdup[idx], 
                     -90*np.ones_like(idx), 
                     yup[idx], 
                     color=colors[np.abs(k)], 
                     alpha=0.5)
plt.show()
plt.ylim(-90,10)
plt.xlim(-30000,30000)

#%%
plt.figure()
plt.plot(v, twPatternAbs[500,:])
plt.grid()
plt.title('Elevation pattern')
plt.xlabel('v')
plt.ylabel('Response (dB)')
plt.show()
         
#%% Plot the two-way pattern
plt.figure()
plt.imshow(20*np.log10(np.abs(tPatt*rPatt)))
plt.colorbar()
plt.grid()
plt.show()

#%% Compute the directivity
pTXav = np.real(np.mean(tPatt*tPatt.conj(), axis=0))
tDirectivity = tPatt.dot(np.diag(1/np.sqrt(pTXav)))
rDirectivity = v.size*rPatt.dot(np.diag(1/np.linalg.norm(rPatt, axis=0)))

#%% Compute the one way patterns in azimuth
tPwr = np.real(np.sum(TX*TX.conj(), axis=0))
rNrm = np.sqrt(np.real(np.sum(RX*RX.conj(), axis=0)))
nmTX = np.diag(1/np.linalg.norm(np.abs(TX), axis=0))
nmRX = np.diag(1/rNrm)
#tPatt = (np.tile(ePattern, (TX.shape[1], 1)).T)*(dM.dot(tTX))#.dot(nmTX)
pTXav = np.real(np.mean(tPatt*tPatt.conj(), axis=0))
tDirectivity = tPatt.dot(np.diag(1/np.sqrt(pTXav)))
#rPatt = (np.tile(ePattern, (RX.shape[1], 1)).T)*(dM.dot(tRX)).dot(nmRX)
rDirectivity = v.size*rPatt.dot(np.diag(1/np.linalg.norm(rPatt, axis=0)))

#%% Plot TX directivity
plt.figure()
plt.plot(v, 20*np.log10(np.abs(tPatt)))
plt.grid()
plt.xlabel("v")
plt.ylabel("Directivity (dB)")
plt.show()

plt.figure()
plt.plot(v, 20*np.log10(np.abs(tDirectivity)))
plt.grid()
plt.xlabel("v")
plt.ylabel("Directivity (dB)")
plt.show()

#%% Write to file
def createXMLStructure(folder): 
    for fd in ["simulation_data", 
               "simulation_plots",
               os.path.join("simulation_data", "raw"),
               os.path.join("simulation_data", "mchan_processed"),
               os.path.join("simulation_data", "wk_processed")]:
        tfd = os.path.join(folder, fd)
        if not os.path.exists(tfd):
            try:
                os.makedirs(tfd)
            except IOError as ie:
                print(ie)
                return False
    return True

#%% Write to file
def writeToXML(xml_file_name, xml):  
    with open(xml_file_name,'w') as outfile:
        outfile.write(etree.tostring(xml, pretty_print=True).decode())
    
#%% Generate the XML object. This code needs refactoring and documentation
def generateXML(vv,
                va = 7500.0,
                nearRange = 5.746681592636996e-03,
                element_length = 0.04):
    
    print("SURE configuration generator")
    numAziSamples = 8192
    numRngSamples = 1024
    
    #%% Generate the base XML object
    pq = etree.XMLParser(remove_blank_text=True)
    xml = etree.XML(base_SentinelNG,parser=pq)
    fc = float(xml.find(".//carrierFrequency").text)
        
    #%% Set some defaults
    azimuthResolution = 11.8 #vv.az_resolution
    rangeResolution = c/2.0/54.5624802e6 #vv.rn_resolution
    range_BANDWIDTH = c/2.0/rangeResolution
    deltaR = 1.0/1.2/range_BANDWIDTH #1.0/vv.rn_oversample/range_BANDWIDTH
    swathWidth = 88e3 #vv.swath_width
    M = 4
        
    number_channels = M + 1
    print("Number of channels: (M+1) = %d" % number_channels)
    print("Ideal PRF: %0.4f Hz" % (va/(M+1)/azimuthResolution))
    
    #%% Compute some collection parameters
    number_beams = 1 #number_channels
    aziDoppler = va/azimuthResolution
    beam_DOPPLER = aziDoppler #/number_channels
    prf = (va/(number_channels)/azimuthResolution)*1.2 #vv.az_oversample
    near_RANGE = nearRange*c/2.0
    far_RANGE = near_RANGE+swathWidth
    channel_LEN = np.int(np.round(far_RANGE*(c/fc)
                                  *(prf/number_channels)
                                  *(beam_DOPPLER)
                                  /2.0/(va**2)))
    print("Doppler bandwidth per beam: %f Hz" % beam_DOPPLER)
    print("Number of samples per beam in aperture: %d" % channel_LEN)
    print("Number of samples/aperture (channels and beams combined): %d" 
          % (channel_LEN*number_channels*number_beams))
    print("Near range: %f m" % near_RANGE)
    print("Far range: %f m" % far_RANGE)
    print("Number of channels: %d" % number_channels)
    print("Number of beams: %d" % number_beams)
    print("Slow time PRF: %f Hz" % prf)
    
    
    #%% Compute the azimuth positions
    # numAziSamples = 2*channel_LEN*(2+number_channels*number_beams)
    
    # subarray_length = number_channels*azimuthResolution*2.0
    
    
    # # Compute the channel tables for each pulse in the g vector. 
    # # i.e. put in the squint
    # subarray_elements = int(subarray_length/element_length)
    # element_spacing = subarray_length/subarray_elements
    # print("Beam spread interval: %f Hz" % beam_DOPPLER)
    # print("Antenna Element Length: %f m" % element_length)
    # print("Subarray Length: %f m" % subarray_length)
    # subarray_centres = [(pos-(number_channels-1)/2.0)*subarray_length 
    #                     for pos in range(number_channels)]
    # azi_positions = [[subpos + element_spacing/2.0 - 
    #                   subarray_length/2.0 + k*element_spacing 
    #                   for k in range(subarray_elements)] 
    #                  for subpos in subarray_centres]
    
    #%% Write the azimuth positions, lengths and powers
    
    # flattened_azi_positions = reduce(lambda x,y: x+y, azi_positions)
    # str_azi_pos = ["%0.6f" % pos for pos in flattened_azi_positions]
    # str_azi_len = ["%0.6f" % element_length 
    #                for pos in flattened_azi_positions]
    # str_azi_pwr = ["%0.1f" % vv.element_power 
    #                for pos in flattened_azi_positions]
    # print_azi=False
    # if print_azi:
    #     print("Azimuth positions:")
    #     print("----------------------------------------------------")
    #     print(" ".join(str_azi_pos))
    # azimuthPosNode = xml.find('.//azimuthPositions')
    # azimuthPosNode.text = " ".join(str_azi_pos)
    # azimuthLenNode = xml.find('.//azimuthElementLengths')
    # azimuthLenNode.text = " ".join(str_azi_len)
    # azimuthPwrNode = xml.find('.//transmitPowers')
    # azimuthPwrNode.text = " ".join(str_azi_pwr)
    
    #%% Calculate the Doppler centroids for each beam
    beam_DC = [beam_DOPPLER*(pos-(number_beams-1)/2.0) 
               for pos in range(number_beams)]
    reducedPRF = prf/len(beam_DC)
    print("Doppler frequencies:")
    print("----------------------------------------------------")
    print(beam_DC)
    txChan = int((number_channels-1)/2)
     
    #%% Generate the reference time from which to compute the offsets. e.t.c.
    reference_TIME = datetime.combine(date(2015,1,1),time(0,0,0))
    sample_TIMES = [reference_TIME + timedelta(seconds=float(s)/prf) 
                    for s in range(len(beam_DC))]
    nref_TIME = np.datetime64("2015-01-01T00:00:00")
    nsample_TIMES = [nref_TIME
                     + np.timedelta64(int(np.round(1.0e9*float(s)/prf)), 'ns') 
                     for s in range(len(beam_DC))]
    
    #%% Calculate how many extra range samples we'll need
    u = c/fc/(4*azimuthResolution)
    sim_range_samples = 2*(far_RANGE - near_RANGE)/c/deltaR
    RMC_range = (far_RANGE/c*2)*(np.sqrt(1.0+u**2)-1)/deltaR + numRngSamples
    
    min_range_samples = int(RMC_range)
    # Add the offset due to range migration to the nim range samples
    print("RCM additional range: %f" % (RMC_range))
    print("Full simulation required range samples: %f" % sim_range_samples)
    print("New computation: %f" 
          % (2.0*far_RANGE/c*(1.0/np.cos(c/fc/azimuthResolution) - 1)/deltaR))
    print("Minumum number of range samples in a pulse: %f" % min_range_samples)
    liked_range_samples = np.array(sorted(
        np.outer(3**(np.arange(5)), 2**(9 + np.arange(10))).flatten()))
    numRngSamples = liked_range_samples[liked_range_samples>min_range_samples][0]
    print("Number of range samples for this simulation: %d" % numRngSamples)
    print("Number of azimuth samples per signal file: %d" 
          % int(numAziSamples/number_beams))
    
    #%% Estimate the size of the data
    numSigElements = number_channels*len(beam_DC)
    fileSize = 16*numRngSamples*numAziSamples/number_beams
    print("Total number of signal elements: %d" % numSigElements)
    print("Size of each signal data file: %d" % int(fileSize))
    print("Size of all signal files in memory: %f (GB)" 
          % float(fileSize*numSigElements/1e9))
    
    predicted_bands = np.arange(-int(numSigElements/2)-2,
                                int(numSigElements/2)+1+2)
    print("Predicted number of bands: %d" % len(predicted_bands))
    mchanFileSize = len(predicted_bands)*fileSize
    print("Predicted size of multi-channel processed file: %f (GB)" 
          % float(mchanFileSize/1e9))
    print("Peak memory requirement (approx): %f (GB)" 
          % float((mchanFileSize+fileSize*numSigElements)/1e9))
    
    #%% Create the mode XML mode mode snippet
    for fd,n,thisTime,isoTime in zip(beam_DC, 
                                     range(len(beam_DC)), 
                                     sample_TIMES, 
                                     nsample_TIMES):
        phases = [[pi*x*fd/va for x in azipos] for azipos in azi_positions]
        truedelay = [[1.0e9*x/(2.0*va)*fd/fc for x in azipos] 
                     for azipos in azi_positions]
                
        for channel in range(number_channels):
            signalData = etree.Element("signalData")
            beam_Angle = -c*fd/(2.0*va*fc)
            
            # Add some nodes
            timeNode = etree.SubElement(signalData,"azimuthStartTime")
            timeNode.set("iso", str(isoTime))
            yearNode = etree.SubElement(timeNode,"year")
            monthNode = etree.SubElement(timeNode,"month")
            dayNode = etree.SubElement(timeNode,"day")
            hourNode = etree.SubElement(timeNode,"hour")
            minuteNode = etree.SubElement(timeNode,"minute")
            secondNode = etree.SubElement(timeNode,"sec")
            usecNode = etree.SubElement(timeNode,"usec")
            systemNode = etree.SubElement(timeNode,"time_system")
    
            yearNode.text = '%0.4d'%thisTime.year
            monthNode.text = '%0.2d'%thisTime.month
            dayNode.text = '%0.2d'%thisTime.day
            hourNode.text = '%0.2d'%thisTime.hour
            minuteNode.text = '%0.2d'%thisTime.minute
            secondNode.text = '%0.2d'%thisTime.second
            usecNode.text = '%0.6d'%thisTime.microsecond
            systemNode.text = '2'
    
            numRangeSamples = etree.SubElement(signalData,"numRangeSamples")
            numRangeSamples.set("unit","Pixels")
            numRangeSamples.text = "%d" % numRngSamples
    
            numAzimuthSamples = etree.SubElement(signalData,
                                                 "numAzimuthSamples")
            numAzimuthSamples.set("unit","Pixels")
            numAzimuthSamples.text = "%d" % (numAziSamples/number_beams)
    
    
            nearRangeTime = etree.SubElement(signalData,"nearRangeTime")
            nearRangeTime.set("unit","s")
            nearRangeTime.text = "%0.10e" % nearRange
    
            rangeSampleSpacing = etree.SubElement(signalData,
                                                  "rangeSampleSpacing")
            rangeSampleSpacing.set("unit","s")
            rangeSampleSpacing.text = "%0.6e" % deltaR
    
            pulseRepetitionFrequency = etree.SubElement(signalData,
                                                        "pulseRepetitionFrequency")
            pulseRepetitionFrequency.set("unit","Hz")
            pulseRepetitionFrequency.text = "%0.4f" % reducedPRF
    
            pulseBandwidth = etree.SubElement(signalData,"pulseBandwidth")
            pulseBandwidth.set("unit","MHz")
            pulseBandwidth.text = "%f" % (range_BANDWIDTH/1.0e6)
    
            pulseDuration = etree.SubElement(signalData,"pulseDuration")
            pulseDuration.set("unit","us")
            pulseDuration.text = "%0.4f" % vv.pulse_duration
            
            # Add the radar configuration
            config = etree.Element("radarConfiguration")
            config.set("channel", "channel%d-%d" % (channel,n))
            Tx = etree.Element("transmitConfiguration")
            Rx = etree.Element("receiveConfiguration")
            Tpol = etree.Element("polarization")
            Tpol.text = "H"
            Rpol = etree.Element("polarization")
            Rpol.text = "H"
            txChanMask = [int(bool(x==txChan)) for x in range(number_beams)]
            txWeight = [[y for x in azi_positions[channel]] 
                        for y in txChanMask]
            txMag = etree.Element("magnitude")
            txMag.set("unit", "natural")
            txMag.text = " ".join(["%d" % x for x in 
                                   reduce(lambda x,y: x+y, txWeight)])
            txPhase = etree.Element("phase")
            txPhase.set("unit", "radians")
            txPhase.text = " ".join(["%0.3f" % x for x in 
                                     reduce(lambda x,y: x+y, phases)])
            txDelay = etree.Element("truedelay")
            txDelay.set("unit", "ns")
            txDelay.text = " ".join(["%0.6f" % x for x in 
                                     reduce(lambda x,y: x+y, truedelay)])
            rxChanMask = [int(bool(x==channel)) for x in range(number_beams)]
            rxWeight = [[y for x in azi_positions[channel]] 
                        for y in rxChanMask]
            rxMag = etree.Element("magnitude")
            rxMag.set("unit", "natural")
            rxMag.text = " ".join(["%d" % x for x in 
                                   reduce(lambda x,y: x+y, rxWeight)])
            rxPhase = etree.Element("phase")
            rxPhase.set("unit", "radians")
            rxPhase.text = " ".join(["%0.3f" % x for x in 
                                     reduce(lambda x,y: x+y, phases)])
            rxDelay = etree.Element("truedelay")
            rxDelay.set("unit", "ns")
            rxDelay.text = " ".join(["%0.6f" % x for x in 
                                     reduce(lambda x,y: x+y, truedelay)])
            txuZero = etree.Element("u0")
            txuZero.set("unit", "radians")
            txuZero.text = "%0.6e" % beam_Angle
            rxuZero = etree.Element("u0")
            rxuZero.set("unit", "radians")
            rxuZero.text = "%0.6e" % beam_Angle
            Tx.append(Tpol)
            Tx.append(txuZero)
            Tx.append(txMag)
            Tx.append(txPhase)
            Tx.append(txDelay)
            Rx.append(Rpol)
            Rx.append(rxuZero)
            Rx.append(rxMag)
            Rx.append(rxPhase)
            Rx.append(rxDelay)
            config.append(Tx)
            config.append(Rx)
            signalData.append(config)
            thisfilename = etree.SubElement(signalData,"filename")
            thisfilename_text = "%s_%s_c%db%d.npy" % (os.path.split(vv.config_xml)[-1].split(".")[0], 
                                                      vv.file_data_domain, 
                                                      channel, 
                                                      n)
            thisfilename.text = os.path.sep.join([os.path.split(vv.config_xml)[0],
                                                  "simulation_data",
                                                  "raw",
                                                  thisfilename_text])
            xml.append(signalData)
            
    return xml

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
def printRadarConfigurations(radarConfigurations):
    for config, index in zip(radarConfigurations, range(len(radarConfigurations))):
        tag = config.get('channel')
        txPol = config.xpath('transmitConfiguration/polarization')
        txCols = config.xpath('transmitConfiguration/magnitude')
        rxPol = config.xpath('receiveConfiguration/polarization')
        rxCols = config.xpath('receiveConfiguration/magnitude')
        print("%0.2d"%(index+1) + ': ' + ' --> ' + txPol[0].text + ': ' + txCols[0].text + ' <-- ' + rxPol[0].text + ': '+ rxCols[0].text + ' ' + tag)

