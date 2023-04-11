#!/usr/bin/env python2.7

""" This python program should generate the config files for burst mode multi or single channel data
    The program generates an xml file to be processed in MATLAB """

import sys
import getopt
import numpy as np
from datetime import datetime, date, time, timedelta
from math import pi
import urllib
from copy import deepcopy
import lxml.etree as etree
import argparse
from scipy.constants import c

#%% Do the argument parsing stuff here
def get_parser():
    description = """ Generate an XML file for multi-channel signal
                      processing.
                      
                      This program sets up also for a burst mode."""
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument("--config-xml", 
                        required=True,
                        help="The name of the xml file to write")
    
    parser.add_argument("--prf",
                        default = 1000.0,
                        type = float)
    
    parser.add_argument("--mode",
                        help="""The mode to use. Choices from:
                                    "MODEX",
                                    "UltraFine",
                                    "Classical"
                                The default choice is MODEX""",
                        choices=["MODEX",
                                 "UltraFine",
                                 "Classical"],
                        default="MODEX")
    
    parser.add_argument("--burst-length",
                        default = 1,
                        type = int,
                        help="Number of Samples in a burst")
    
    parser.add_argument("--centroids",
                        help="Doppler centroids for beams",
                        nargs = "*",
                        default = [0.0],
                        type = float)
    
    parser.add_argument("--num-azimuth-samples",
                        default = 8192,
                        type = int)
    
    parser.add_argument("--range-samples",
                        help=""""Instead of simulating the entire swath, 
                                 use fewer samples. Subset of the swath (int)""",
                        type = int,
                        default = 1024)
    
    
    parser.add_argument("--rn-resolution",
                        help="The desired range resolution (m)",
                        type = float,
                        default = 5.0)
    parser.add_argument("--swath-width",
                        help="The desired swath width (m)",
                        type = float,
                        default = 10000.0)
    parser.add_argument("--rn-oversample",
                        help="The desired range oversample factor",
                        type = float,
                        default = 1.2)
    # parser.add_argument("--az-oversample",
    #                     help="The desired azimuth oversample factor",
    #                     type = float,
    #                     default = 1.2)
    parser.add_argument("--pulse-duration",
                        help="Desired pulse duration in us",
                        type = float,
                        default = 20.5)

    return parser

def main(argv):
    print("Wiper burst configuration generator")
    parser = get_parser()
    vv = parser.parse_args()
    
    prf = vv.prf
    numAziSamples = vv.num_azimuth_samples
    burst_LENGTH = vv.burst_length 
    p = etree.XMLParser(remove_blank_text=True)
    base_XML = """
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
            <transmitPowers unit="dB">30 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30</transmitPowers>
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

    # Generate the base XML object
    pq = etree.XMLParser(remove_blank_text=True)
    xml = etree.XML(base_XML,parser=pq)
    azimuthPosNode = xml.findall('.//azimuthPositions')
    xpos = [float(pos) for pos in azimuthPosNode[0].text.split()]

    beam_DOPPLER = vv.centroids
    modeSnippet1 = """
    <mode>
        <radarConfiguration channel="MODEX-1 Aft HH">
            <transmitConfiguration>
                <polarization>H</polarization>
                <magnitude unit="natural">1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1</magnitude>
                <phase unit="degrees">0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0</phase>
            </transmitConfiguration>
            <receiveConfiguration>
                <polarization>H</polarization>
                <magnitude unit="natural">1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0</magnitude>
                <phase unit="degrees">0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0</phase>
            </receiveConfiguration>
        </radarConfiguration>
        <radarConfiguration channel="MODEX-1 Fore HH">
            <transmitConfiguration>
                <polarization>H</polarization>
                <magnitude unit="natural">1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1</magnitude>
                <phase unit="degrees">0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0</phase>
            </transmitConfiguration>
            <receiveConfiguration>
                <polarization>H</polarization>
                <magnitude unit="natural">0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1</magnitude>
                <phase unit="degrees">0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0</phase>
            </receiveConfiguration>
        </radarConfiguration>
    </mode>
    """
    
    modeSnippet2 = """
    <mode>
        <radarConfiguration channel="UltraFine Aft HH">
	  <transmitConfiguration>
	    <polarization>H</polarization>
	    <magnitude unit="natural">1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1</magnitude>
	    <phase unit="radians">1.963 1.374 1.374 0.982 0.785 0.393 0.000 0.000 0.000 0.000 0.393 0.785 0.982 1.374 1.374 1.963</phase>
	  </transmitConfiguration>
	  <receiveConfiguration>
	    <polarization>H</polarization>
	    <magnitude unit="natural">0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0</magnitude>
	    <phase unit="radians">0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0</phase>
	  </receiveConfiguration>
        </radarConfiguration>								
        <radarConfiguration channel="UltraFine Fore HH">
	  <transmitConfiguration>
	    <polarization>H</polarization>
	    <magnitude unit="natural">1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1</magnitude>
	    <phase unit="radians">1.963 1.374 1.374 0.982 0.785 0.393 0.000 0.000 0.000 0.000 0.393 0.785 0.982 1.374 1.374 1.963</phase>
	  </transmitConfiguration>
	  <receiveConfiguration>
	    <polarization>H</polarization>
	    <magnitude unit="natural">0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0</magnitude>
	    <phase unit="radians">0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0</phase>
	  </receiveConfiguration>
        </radarConfiguration>	
    </mode>
    """
    
    modeSnippet3 = """
    <mode>
        <radarConfiguration channel="Full Antenna">
            <transmitConfiguration>
                <polarization>H</polarization>
                <magnitude unit="natural">1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1</magnitude>
                <phase unit="degrees">0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0</phase>
            </transmitConfiguration>
            <receiveConfiguration>
                <polarization>H</polarization>
                <magnitude unit="natural">1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1</magnitude>
                <phase unit="degrees">0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0</phase>
            </receiveConfiguration>
        </radarConfiguration>
    </mode>
    """
    modeSnippet={'MODEX': modeSnippet1, 'UltraFine': modeSnippet2, 'Classical': modeSnippet3}
    
    va = 7500.0
    output_FILE = vv.config_xml
    #radarConfigURL = "/home/sikaneta/Documents/Radarsat_Modes/radarConfiguration.xml"
    #modeChoice = 'MODEX'

    # Load the configuration file
    # try:
    #     opts, args = getopt.getopt(argv,"hp:l:f:a:d:m:",["help","prf=","burst_samples=","xmlfile=","numAzimuthSamples=","dopplerSpread=","mode="])
    # except getopt.GetoptError as err:
    #     print('Options error')
    #     print(err)
    #     print('generateBurstConfig -p <--prf=> -l <--burst_samples=> -f <--xmlfile=> -a <--numAzimuthSamples> -d <--dopplerSpread> -m <--mode>')
    #     sys.exit(2)

    # for opt, arg in opts:
    #     if opt in ("-h", "--help"):
    #         print('generateBurstConfig -p <--prf=> -l <--burst_samples=> -f <--xmlfile=>')
    #     elif opt in ("-p", "--prf"):
    #         prf = float(arg)
    #     elif opt in ("-l", "--burst_samples"):
    #         burst_LENGTH = int(arg)
    #     elif opt in ("-d", "--dopplerSpread"):
    #         #beam_DOPPLER = [-float(arg),float(arg)]
    #         beam_DOPPLER = [int(k) for k in arg.split(',')]
    #     elif opt in ("-f", "--xmlfile"):
    #         output_FILE = arg
    #     elif opt in ("-a", "--numAzimuthSamples"):
    #         numAziSamples = int(arg)
    #     elif opt in ("-m", "--mode"):
    #         if(arg in modeSnippet):
    #             modeChoice = arg
    #         else:
    #             print('(E) Error: Mode not recognized')
    #             print('Choose from:')
    #             print('------------')
    #             for key in modeSnippet.keys():
    #                 print(key)
    #             sys.exit(2)

    # Read the mode XML snippet
    modeXML = [etree.XML(modeSnippet[vv.mode],parser=p) for doppler in beam_DOPPLER]
    
    # Compute the channel tables for each pulse in the g vector. i.e. put in the squint
    allModes = []
    print(beam_DOPPLER)
    for mode,fd in zip(modeXML,beam_DOPPLER):
        channels = mode.findall("radarConfiguration")
        for channel in channels:
            tx = channel.findall('transmitConfiguration')
            phase = tx[0].findall('phase')
            if(phase[0].get("unit")=="degrees"):
                phases = [float(k)+180*x*fd/va for k,x in zip(phase[0].text.split(),xpos)]
            elif(phase[0].get("unit")=="radians"):
                phases = [float(k)+pi*x*fd/va for k,x in zip(phase[0].text.split(),xpos)]
            phase[0].text = " ".join(["%0.4f" % p for p in phases])
            rx = channel.findall('receiveConfiguration')
            phase = rx[0].findall('phase')
            if(phase[0].get("unit")=="degrees"):
                phases = [float(k)+180*x*fd/va for k,x in zip(phase[0].text.split(),xpos)]
            elif(phase[0].get("unit")=="rad"):
                phases = [float(k)+pi*x*fd/va for k,x in zip(phase[0].text.split(),xpos)]
            phase[0].text = " ".join(["%0.4f" % p for p in phases])
        allModes.append(channels)

    # Generate the reference time from which to compute the offsets. e.t.c.
    reference_TIME = datetime.combine(date(2015,1,1),time(0,0,0))
    
    # Now generate the start times
    n_BEAMS = len(beam_DOPPLER)
    sample_TIMES = [[reference_TIME + timedelta(seconds=float(s+burst_LENGTH*beam)/(prf)) for s in range(burst_LENGTH)] for beam in range(n_BEAMS)]
    # sample_TIMES = [[reference_TIME + timedelta(seconds=float(s)/prf) for mode in modeXML] for s in range(burst_LENGTH*n_BEAMS)]

    for tm,mode in zip(sample_TIMES,allModes):
        for thisTime in tm:
            for config in mode:
                # Add the signal elements
                signalData = etree.SubElement(xml,"signalData")
                timeNode = etree.SubElement(signalData,"azimuthStartTime")
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
                numRangeSamples.text = "%d" % vv.range_samples

                numAzimuthSamples = etree.SubElement(signalData,"numAzimuthSamples")
                numAzimuthSamples.set("unit","Pixels")
                numAzimuthSamples.text = "%d" % (numAziSamples/(n_BEAMS*burst_LENGTH))


                nearRangeTime = etree.SubElement(signalData,"nearRangeTime")
                nearRangeTime.set("unit","s")
                nearRangeTime.text = "5.568856572102291e-03"

                rangeSampleSpacing = etree.SubElement(signalData,"rangeSampleSpacing")
                rangeSampleSpacing.set("unit","s")
                rangeSampleSpacing.text = "%0.8e" % (2*vv.rn_resolution/(vv.rn_oversample*c))

                pulseRepetitionFrequency = etree.SubElement(signalData,"pulseRepetitionFrequency")
                pulseRepetitionFrequency.set("unit","Hz")
                pulseRepetitionFrequency.text = "%0.4f" % (prf/float(n_BEAMS*burst_LENGTH))

                pulseBandwidth = etree.SubElement(signalData,"pulseBandwidth")
                pulseBandwidth.set("unit","MHz")
                pulseBandwidth.text = "%0.8f" % (c/(2*vv.rn_resolution)/1e6)

                pulseDuration = etree.SubElement(signalData,"pulseDuration")
                pulseDuration.set("unit","us")
                pulseDuration.text = "%f" % vv.pulse_duration

                signalData.append(deepcopy(config))

    with open(output_FILE,'w') as outfile:
        outfile.write(etree.tostring(xml, pretty_print=True).decode())


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

def printRadarConfigurations(radarConfigurations):
    for config, index in zip(radarConfigurations, range(len(radarConfigurations))):
        tag = config.get('channel')
        txPol = config.xpath('transmitConfiguration/polarization')
        txCols = config.xpath('transmitConfiguration/magnitude')
        rxPol = config.xpath('receiveConfiguration/polarization')
        rxCols = config.xpath('receiveConfiguration/magnitude')
        print("%0.2d"%(index+1) + ': ' + ' --> ' + txPol[0].text + ': ' + txCols[0].text + ' <-- ' + rxPol[0].text + ': '+ rxCols[0].text + ' ' + tag)






if __name__=="__main__":
    main(sys.argv[1:])
