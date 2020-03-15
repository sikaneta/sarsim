#!/usr/bin/env python2.7

#%%
""" This python program should generate the config files for burst mode multi or single channel data
    The program generates an xml file to be processed in MATLAB """

import numpy as np
from datetime import datetime, date, time, timedelta
from math import pi
import urllib
from copy import deepcopy
import lxml.etree as etree
from functools import reduce
import os

#%% Write to file
def createXMLStructure(folder): 
    for fd in ["l0_data", "l1_data", "l2_data"]:
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
def generateXML(vv):
    print("SURE configuration generator")
    numAziSamples = 8192
    burst_LENGTH = 4 
    p = etree.XMLParser(remove_blank_text=True)
    
    base_XML_CBand = """
    <gmtiAuxDataParams schemaVersion="1.0">
      <acquisitionParameters>
        <mode>Ish_Burst_Mode</mode>
        <instrument>
          <antennaArray>     
            <carrierFrequency unit="Hz">5.405e9</carrierFrequency>
            <azimuthPositions unit="m">0.4688 1.4062 2.3438 3.2812 4.2188 5.1562 6.0938 7.0312 7.9688 8.9062 9.8438 10.7812 11.7188 12.6562 13.5938 14.5312</azimuthPositions>
            <elevationPositions unit="m">0.75</elevationPositions>
            <azimuthElementLengths unit="m">0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286</azimuthElementLengths>
            <elevationElementLengths unit="m">1.5</elevationElementLengths>
            <transmitPowers unit="dB">1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1</transmitPowers>
            <systemTemperature unit="degrees" system="Kelvin">297</systemTemperature>
            <systemLosses unit="dB">-5.3</systemLosses>
          </antennaArray>
                      
          <lookDirection>Right</lookDirection>
          <orbit>
            <inclination unit="degrees">98.6</inclination>
            <eccentricity>0.00</eccentricity>
            <semiMajorAxis unit="m">7071009</semiMajorAxis>
            <angleOfPerigee unit="degrees">90.0</angleOfPerigee>
            <orbitAngle unit="degrees">60</orbitAngle>
          </orbit>
        </instrument>
        <platformLongitude unit="degrees" referenceEllipsoid="WGS84">90</platformLongitude>
      </acquisitionParameters>
    </gmtiAuxDataParams>"""
    
    base_XML_XBand = """
    <gmtiAuxDataParams schemaVersion="1.0">
      <acquisitionParameters>
        <mode>Ish_Burst_Mode</mode>
        <instrument>
          <antennaArray>     
            <carrierFrequency unit="Hz">9.650e9</carrierFrequency>
            <azimuthPositions unit="m">0.4688 1.4062 2.3438 3.2812 4.2188 5.1562 6.0938 7.0312 7.9688 8.9062 9.8438 10.7812 11.7188 12.6562 13.5938 14.5312</azimuthPositions>
            <elevationPositions unit="m">0.75</elevationPositions>
            <azimuthElementLengths unit="m">0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286 0.9286</azimuthElementLengths>
            <elevationElementLengths unit="m">1.5</elevationElementLengths>
            <transmitPowers unit="dB">1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1</transmitPowers>
            <systemTemperature unit="degrees" system="Kelvin">297</systemTemperature>
            <systemLosses unit="dB">-5.3</systemLosses>
          </antennaArray>
                      
          <lookDirection>Right</lookDirection>
          <orbit>
            <inclination unit="degrees">98.6</inclination>
            <eccentricity>0.00</eccentricity>
            <semiMajorAxis unit="m">7071009</semiMajorAxis>
            <angleOfPerigee unit="degrees">90.0</angleOfPerigee>
            <orbitAngle unit="degrees">60</orbitAngle>
          </orbit>
        </instrument>
        <platformLongitude unit="degrees" referenceEllipsoid="WGS84">90</platformLongitude>
      </acquisitionParameters>
    </gmtiAuxDataParams>"""
    
    #%% Generate the base XML object
    pq = etree.XMLParser(remove_blank_text=True)
    xml = etree.XML(base_XML_XBand,parser=pq)
    fc = float(xml.find(".//carrierFrequency").text)
        
    #%% Set some defaults
    c = 299792458.0
    va = 7500.0
    azimuthResolution = vv.az_resolution
    rangeResolution = vv.rn_resolution
    range_BANDWIDTH = c/2.0/rangeResolution
    range_OVERSAMPLING = vv.rn_oversample
    nearRange = 5.568856572102291e-03
    deltaR = 1.0/range_OVERSAMPLING/range_BANDWIDTH
    swathWidth = vv.swath_width
    #numRngSamples = swathWidth/(deltaR*c/2.0)
    rangeTimeBufferFactor = 2.0
    min_M = int(np.ceil(rangeTimeBufferFactor*va*2.0*swathWidth/azimuthResolution/c - 1)) 
    max_M = int(np.floor(np.sqrt(vv.max_antenna_length/azimuthResolution/2) - 1))
    if max_M >= min_M:
        M = max_M
    else:
        print("No solution for the number of channels")
        
    number_channels = M + 1
    print("Number of channels: (M+1) = %d" % number_channels)
    print("Antenna total length: %0.2f m" % (2.0*azimuthResolution*(M+1)**2))
    print("Sub-aperture length: %0.2f m" % (2.0*azimuthResolution*(M+1)))
    print("Ideal PRF: %0.4f Hz" % (va/(M+1)/azimuthResolution))
    #%%   
    number_beams = number_channels
    aziDoppler = 0.9*va/azimuthResolution
    beam_DOPPLER = aziDoppler/number_channels
    #azimuth_OVERSAMPLING = 1.0
    prf = (va/(number_channels)/azimuthResolution)
    near_RANGE = nearRange*c/2.0
    far_RANGE = near_RANGE+swathWidth
    sample_FACTOR = 1.5
    channel_LEN = np.int(np.round(sample_FACTOR*far_RANGE*(c/fc)*(prf/number_channels)*(beam_DOPPLER)/2.0/(va**2)))
    print("Doppler bandwidth per beam: %f Hz" % beam_DOPPLER)
    print("Number of samples per beam in aperture: %d" % channel_LEN)
    print("Number of samples/aperture (channels and beams combined): %d" % (channel_LEN*number_channels*number_beams))
    print("Near range: %f m" % near_RANGE)
    print("Far range: %f m" % far_RANGE)
    print("Number of channels: %d" % number_channels)
    print("Number of beams: %d" % number_beams)
    print("Slow time PRF: %f Hz" % prf)
    
    
    #%%
    numAziSamples = 2*channel_LEN*(2+number_channels*number_beams)
    
    element_length = 0.04
    #subarray_length = 2.2
    subarray_length = number_channels*azimuthResolution*2.0
    
    
    # Compute the channel tables for each pulse in the g vector. i.e. put in the squint
    #allModes = []
    subarray_elements = int(subarray_length/element_length)
    element_spacing = subarray_length/subarray_elements
    print("Beam spread interval: %f Hz" % beam_DOPPLER)
    print("Antenna Element Length: %f m" % element_length)
    print("Subarray Length: %f m" % subarray_length)
    subarray_centres = [(pos-(number_channels-1)/2.0)*subarray_length for pos in range(number_channels)]
    azi_positions = [[subpos + element_spacing/2.0 - subarray_length/2.0 + k*element_spacing for k in range(subarray_elements)] for subpos in subarray_centres]
    
    #%%
    
    flattened_azi_positions = reduce(lambda x,y: x+y, azi_positions)
    str_azi_pos = ["%0.6f" % pos for pos in flattened_azi_positions]
    str_azi_len = ["%0.6f" % element_length for pos in flattened_azi_positions]
    str_azi_pwr = ["%0.1f" % vv.element_power for pos in flattened_azi_positions]
    print("Azimuth positions:")
    print("----------------------------------------------------")
    print(" ".join(str_azi_pos))
    azimuthPosNode = xml.find('.//azimuthPositions')
    azimuthPosNode.text = " ".join(str_azi_pos)
    azimuthLenNode = xml.find('.//azimuthElementLengths')
    azimuthLenNode.text = " ".join(str_azi_len)
    azimuthPwrNode = xml.find('.//transmitPowers')
    azimuthPwrNode.text = " ".join(str_azi_pwr)
    
    # Calculate the Doppler centroids for each beam
    beam_DC = [beam_DOPPLER*(pos-(number_beams-1)/2.0) for pos in range(number_beams)]
    reducedPRF = prf/len(beam_DC)
    print("Doppler frequencies:")
    print("----------------------------------------------------")
    print(beam_DC)
    txChan = int((number_channels-1)/2)
     
    # Generate the reference time from which to compute the offsets. e.t.c.
    reference_TIME = datetime.combine(date(2015,1,1),time(0,0,0))
    sample_TIMES = [reference_TIME + timedelta(seconds=float(s)/prf) for s in range(len(beam_DC))]
    nref_TIME = np.datetime64("2015-01-01T00:00:00")
    nsample_TIMES = [nref_TIME
                     + np.timedelta64(int(np.round(1.0e9*float(s)/prf)), 'ns') for s in range(len(beam_DC))]
    
    # Calculate how many extra range samples we'll need
    u = c/fc/(4*azimuthResolution)
    # RMC_range = (far_RANGE/c*2)/2*(c/fc/(4*azimuthResolution))**2/deltaR
    sim_range_samples = 2*(far_RANGE - near_RANGE)/c/deltaR
    RMC_range = (far_RANGE/c*2)*(np.sqrt(1.0+u**2)-1)/deltaR + vv.range_samples
    
    min_range_samples = int(RMC_range)
    # Add the offset due to range migration to the nim range samples
    print("RCM additional range: %f" % (RMC_range))
    print("Full simulation required range samples: %f" % sim_range_samples)
    print("New computation: %f" % (2.0*far_RANGE/c*(1.0/np.cos(c/fc/azimuthResolution) - 1)/deltaR))
    print("Minumum number of range samples in a pulse: %f" % min_range_samples)
    liked_range_samples = np.array(sorted(np.outer(3**(np.arange(5)), 2**(9 + np.arange(10))).flatten()))
    numRngSamples = liked_range_samples[liked_range_samples>min_range_samples][0]
    print("Number of range samples for this simulation: %d" % numRngSamples)
    
    # Create the mode XML mode mode snippet
    for fd,n,thisTime,isoTime in zip(beam_DC, range(len(beam_DC)), sample_TIMES, nsample_TIMES):
        phases = [[pi*x*fd/va for x in azipos] for azipos in azi_positions]
        truedelay = [[1.0e9*x/(2.0*va)*fd/fc for x in azipos] for azipos in azi_positions]
                
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
            #numRangeSamples.text = "%d" % int(numRngSamples*(1.0+np.abs(beam_Angle)))
    
            numAzimuthSamples = etree.SubElement(signalData,"numAzimuthSamples")
            numAzimuthSamples.set("unit","Pixels")
            numAzimuthSamples.text = "%d" % (numAziSamples/number_beams)
    
    
            nearRangeTime = etree.SubElement(signalData,"nearRangeTime")
            nearRangeTime.set("unit","s")
            nearRangeTime.text = "%0.10e" % nearRange
            #nearRangeTime.text = "%0.10e" % (nearRange + int(nearRange*np.abs(beam_Angle)/deltaR)*deltaR)
    
            rangeSampleSpacing = etree.SubElement(signalData,"rangeSampleSpacing")
            rangeSampleSpacing.set("unit","s")
            rangeSampleSpacing.text = "%0.6e" % deltaR
    
            pulseRepetitionFrequency = etree.SubElement(signalData,"pulseRepetitionFrequency")
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
            txWeight = [[y for x in azi_positions[channel]] for y in txChanMask]
            txMag = etree.Element("magnitude")
            txMag.set("unit", "natural")
            txMag.text = " ".join(["%d" % x for x in reduce(lambda x,y: x+y, txWeight)])
            txPhase = etree.Element("phase")
            txPhase.set("unit", "radians")
            txPhase.text = " ".join(["%0.3f" % x for x in reduce(lambda x,y: x+y, phases)])
            txDelay = etree.Element("truedelay")
            txDelay.set("unit", "ns")
            txDelay.text = " ".join(["%0.6f" % x for x in reduce(lambda x,y: x+y, truedelay)])
            rxChanMask = [int(bool(x==channel)) for x in range(number_beams)]
            rxWeight = [[y for x in azi_positions[channel]] for y in rxChanMask]
            rxMag = etree.Element("magnitude")
            rxMag.set("unit", "natural")
            rxMag.text = " ".join(["%d" % x for x in reduce(lambda x,y: x+y, rxWeight)])
            rxPhase = etree.Element("phase")
            rxPhase.set("unit", "radians")
            rxPhase.text = " ".join(["%0.3f" % x for x in reduce(lambda x,y: x+y, phases)])
            rxDelay = etree.Element("truedelay")
            rxDelay.set("unit", "ns")
            rxDelay.text = " ".join(["%0.6f" % x for x in reduce(lambda x,y: x+y, truedelay)])
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
                                                  "l0_data", 
                                                  thisfilename_text])
            xml.append(signalData)
            
    return xml

#%% Modify configurations
def modifyRadarConfigurations(gmtiDataPool, radarConfigURL, directory):
    # Open the radar configuration file and parse
    radarConfig = etree.parse(urllib.urlopen(radarConfigURL))
    radarConfigurations = radarConfig.xpath('//radarConfiguration')

    # Define the default pulse type
    pulse = {'pulseId': "-", 'pulseString': "Unknown", 'pulseDuration': "0", 'pulseBwd': "0"}

    # Get all the signalData elements
    currentConfigs = gmtiDataPool.xpath('//signalData')

    print("Current configurations are: ")
    printSignalRadarConfigurations(currentConfigs)
    currentConfig = int(raw_input("Enter the number of the configuration you wish to replace/add (0 for none)? "))
    while((currentConfig > 0) and (currentConfig <= len(currentConfigs))):
        targetConfig = int(raw_input("Enter the number of the desired radar configuration (-1 for a list)?"))
        while((targetConfig < 1) or (targetConfig > len(radarConfigurations))):
            printRadarConfigurations(radarConfigurations)
            targetConfig = int(raw_input("Enter the number of the configuration you wish to replace it with (-1 for a list)? "))

        # Rename the element so we dont need to write so much
        configParent = currentConfigs[currentConfig-1]

        # Find and remove the current configuration if it is present
        toRemove = configParent.xpath('radarConfiguration')
        if(len(toRemove)>0):
            # record the current pulse type
            pulse = {'pulseId': configParent.xpath('radarConfiguration/pulse/pulseId')[0].text,
                     'pulseString': configParent.xpath('radarConfiguration/pulse/pulseString')[0].text,
                     'pulseDuration': configParent.xpath('radarConfiguration/pulse/pulseDuration')[0].text,
                     'pulseBwd': configParent.xpath('radarConfiguration/pulse/pulseBwd')[0].text}

            # Remove the old element
            configParent.remove(toRemove[0])
        else:
            dataFile = configParent.xpath('dataFile')
            if(len(dataFile)>0):
                pulse = getPulseType(os.path.join(directory,dataFile[0].text).replace('SarImage_0.tif','PARM_IngestParameters.self'))

        # Add the new config. First locate the point where it will go
        # The schema says there should be a dopplerCentroid element
        dopEl = configParent.find('dopplerCentroid')
        dopEl.addprevious(deepcopy(radarConfigurations[targetConfig-1]))

        # Now add the pulse element
        rConf = configParent.find('radarConfiguration')
        pulseElement = etree.Element("pulse")
        pulseId = etree.SubElement(pulseElement, "pulseId")
        pulseSt = etree.SubElement(pulseElement, "pulseString")
        pulseDu = etree.SubElement(pulseElement, "pulseDuration")
        pulseBw = etree.SubElement(pulseElement, "pulseBwd")
        pulseId.text = pulse['pulseId']
        pulseSt.text = pulse['pulseString']
        pulseDu.text = pulse['pulseDuration']
        pulseBw.text = pulse['pulseBwd']
        rConf.insert(0, pulseElement)

        #currentConfigs[currentConfig-1].getparent().remove(currentConfigs[currentConfig-1])
        currentConfigs = gmtiDataPool.xpath('//signalData')

        # Ask for the next config to change
        print("Current configurations are: ")
        printSignalRadarConfigurations(currentConfigs)
        currentConfig = int(raw_input("Enter the number of the configuration you wish to change (0 for none)? "))

#%% Print configurations
def printRadarConfigurations(radarConfigurations):
    for config, index in zip(radarConfigurations, range(len(radarConfigurations))):
        tag = config.get('channel')
        txPol = config.xpath('transmitConfiguration/polarization')
        txCols = config.xpath('transmitConfiguration/magnitude')
        rxPol = config.xpath('receiveConfiguration/polarization')
        rxCols = config.xpath('receiveConfiguration/magnitude')
        print("%0.2d"%(index+1) + ': ' + ' --> ' + txPol[0].text + ': ' + txCols[0].text + ' <-- ' + rxPol[0].text + ': '+ rxCols[0].text + ' ' + tag)

