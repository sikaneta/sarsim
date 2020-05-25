#%% imports and constants
import xml.etree.ElementTree as etree
import numpy as np
import numpy.matlib as npmat
import datetime as dt
import matplotlib.pyplot as plt
from measurement.measurement import state_vector
from measurement.arclength import slow
from geoComputer.geoComputer import satGeometry as sG
from common.utils import FFT_freq
import utils.fileio as fio
import omegak.omegak as wk
import omegak.nbomegak as nbwk
import omegak.omegakappa as wkap
import omegak.stolz as stlz
from scipy.constants import Boltzmann, c
from scipy import interpolate
import os
import pickle
from numba import cuda
from tqdm import tqdm
from antenna.pattern import numberWaveformComputationSamples as nWCS
if cuda.is_available():
    from antenna.pattern import antennaResponseCudaMem as antennaResp
else:
    from antenna.pattern import antennaResponseMultiCPU as antennaResp

homedir = 'E:\\' #os.environ['HOMEPATH']
defaultConfig = u'E:\\Python\\myrdr2\\radar\\XML\\sureConfig.xml'

class physical:
    c = 299792458.0
    mass_EARTH = 5.97219e24
    G = 6.67384e-11
    
#%matplotlib notebook

#%% Utitlity string converter class for XML data
class stringConverters:
    distu = {'km': lambda x: 1.0e3*x, 
             'm': lambda x: x, 
             'cm': lambda x: x*1.0e-2, 
             'mm': lambda x: x*1.0e-3, 
             'um': lambda x: x*1.0e-6, 
             'nm': lambda x:x*1.0e-9}
    timeu = {'s': lambda x: x, 
             'ms': lambda x: x*1.0e-3, 
             'us': lambda x: x*1.0e-6, 
             'ns': lambda x: x*1.0e-9}
    frequ = {'hz': lambda x: x, 
             'khz': lambda x: x*1.0e3, 
             'mhz': lambda x: x*1.0e6, 
             'ghz': lambda x: x*1.0e9}
    poweru = {'db': lambda x: 10**(x/10.0),
              'natural': lambda x: x}
    
    def getXMLElement(self, fieldName, xmlroot):
        dummy = xmlroot.find(fieldName)
        if dummy is not None:
            return xmlroot.find(fieldName)
        else:
            return None
        
    def readXMLField(self, fieldName, xmlroot, fn):
        dummy = xmlroot.find(fieldName)
        if dummy is not None:
            return fn(xmlroot.find(fieldName).text)
        else:
            return None
        
    def toString(self, x, xmlroot):
        return self.readXMLField(x, xmlroot, lambda x: x)

    def toInt(self, x, xmlroot):
        return self.readXMLField(x, xmlroot, int)
    
    def toDouble(self, x, xmlroot):
        return self.readXMLField(x, xmlroot, float)
    
    def toFrequency(self, x, xmlroot):
        el = self.getXMLElement(x, xmlroot)
        if el is None:
            return None
        try:
            return self.frequ[el.get('unit').lower()](float(el.text))
        except KeyError as ke:
            return float(el.text)
            
    def listToScalar(self, x, xmlroot, fn):
        dummy = fn(x, xmlroot)
        if dummy is None:
            return None
        if len(dummy) == 0:
            return None
        return dummy[0]
    
    def toPower(self, x, xmlroot):
        return self.listToScalar(x, xmlroot, self.toPowerArray)
    
    def toPowerArray(self, x, xmlroot):
        el = self.getXMLElement(x, xmlroot)
        if el is None:
            return None
        try:
            return np.array([self.poweru[el.get('unit').lower()](float(elt)) for elt in el.text.split()])
        except KeyError as ke:
            return np.array([float(elt) for elt in el.text.split()])
            
    def toAngleArray(self, x, xmlroot):
        el = self.getXMLElement(x, xmlroot)
        if el is None:
            return None
        if el.get('unit') in ['degrees', 'Degrees']:
            return np.array([float(elt)/180.0*np.pi for elt in el.text.split()])
        elif el.get('unit') in ['radians', 'Radians']:
            return np.array([float(elt) for elt in el.text.split()])
        else:
            return np.array([float(elt)/180.0*np.pi for elt in el.text.split()])
        
    def toDistance(self, x, xmlroot):
        return self.listToScalar(x, xmlroot, self.toDistanceArray)   
    
    def toDuration(self, x, xmlroot):
        return self.listToScalar(x, xmlroot, self.toDurationArray)
    
    def toAngle(self, x, xmlroot):
        return self.listToScalar(x, xmlroot, self.toAngleArray)
        
    def toDistanceArray(self, x, xmlroot):
        el = self.getXMLElement(x, xmlroot)
        if el is None:
            return None
        try:
            return np.array([self.distu[el.get('unit').lower()](float(elt)) for elt in el.text.split()])
        except KeyError as ke:
            try:
                return np.array([self.timeu[el.get('unit').lower()](float(elt))*physical.c for elt in el.text.split()])
            except KeyError as kte:
                return np.array([float(elt) for elt in el.text.split()])
       
    def toDurationArray(self, x, xmlroot):
        el = self.getXMLElement(x, xmlroot)
        if el is None:
            return None
        try:
            return np.array([self.timeu[el.get('unit').lower()](float(elt)) for elt in el.text.split()])
        except KeyError as ke:
            try:
                return np.array([self.distu[el.get('unit').lower()](float(elt))/physical.c for elt in el.text.split()])
            except KeyError as kte:
                return np.array([float(elt) for elt in el.text.split()])
    
    def toDateTime(self, x, xmlroot):
        targetElement = self.getXMLElement(x, xmlroot)
        if targetElement is not None:
            if targetElement.get("iso") is not None:
                return np.datetime64(targetElement.get("iso"))
            else:
                dummy = dt.datetime(self.toInt('.//year', targetElement),
                               self.toInt('.//month', targetElement),
                               self.toInt('.//day', targetElement),
                               self.toInt('.//hour', targetElement),
                               self.toInt('.//minute', targetElement),
                               self.toInt('.//sec', targetElement),
                               self.toInt('.//usec', targetElement))
            return np.datetime64(dt.datetime.strftime(dummy, "%Y-%m-%dT%H:%M:%S.%f"))
        else:
            return None
        
        
#%% The configuration XML file from pickle file if possible
def loadConfiguration(configFile = None):
    # Make sure there is a config file
    configFile = configFile or defaultConfig
    pkl_file = ".".join(configFile.split(".")[0:-1]) + "_radar.pickle"
    if os.path.exists(pkl_file):
        print("Loading radar metadata from pickle file...")
        with open(pkl_file, 'rb') as f:
           radar = pickle.load(f)
    else:
        print("Loading radar metadata from raw calculations...(takes a while)")
        radar = loadConfigurationRaw(configFile)
        print("Writing radar metadata to pickle file...")
        with open(pkl_file, 'wb') as f:
            pickle.dump(radar, f, pickle.HIGHEST_PROTOCOL)
    return radar
    
#%% Read the configuration xml file   
def loadConfigurationRaw(configFile = None):
    # Make sure there is a config file
    configFile = configFile or defaultConfig
    
    # Get the root Element
    xmlroot = etree.parse(configFile).getroot()
    
    (platform,antenna) = readPlatformParametersElement(xmlroot.find('.//acquisitionParameters'))
    radar = [readSignalDataElement(sigData, antenna) for sigData in xmlroot.findall('.//signalData')]
    
#    refTime = dt.datetime(2000,1,1,0,0,0)
#    secs2K = [(rd['acquisition']['startTime']- refTime).total_seconds() for rd in radar]
#    
    refTime = radar[0]['acquisition']['startTime']
    secs2K = [(rd['acquisition']['startTime']- refTime)/np.timedelta64(1,'s') for rd in radar]
    
    min2K = min(secs2K)
    sdelay = [s2K-min2K for s2K in secs2K]
    for rd, sd in zip(radar, sdelay):
        rd['delay']['sampling'] = sd
    
    # Calculate the time for the state vector calculation
    ref = radar[0]['acquisition']
    np_prf = np.timedelta64(int(np.round(1e9/ref['prf'])),'ns')
    svTime = ref['startTime'] + np_prf*ref['numAzimuthSamples']/2.0
    #rng = (ref['nearRangeTime'] + ref['numRangeSamples']*ref['rangeSampleSpacing'])*physical.c/2.0
    
    platform['orbit']['period'] = np.sqrt(4.0*np.pi**2/(physical.G*physical.mass_EARTH)*platform['orbit']['semiMajorAxis']**3)
    
    # Compute a state vector or read from XML file
    platform['stateVectors'] = orbit2state(platform['orbit'], platform['longitude'], svTime)
    platform['satelliteVelocity'] = np.linalg.norm(platform['stateVectors'].measurementData[0][3:])
    
    # Compute the broadside position at mid-range, mid-azimuth
    for rd in radar:
        rd['platform'] = platform
        rd['antenna'] = antenna
    # acq['satelliteVelocity'] = np.sqrt(sum(acq['stateVectors']['velocity']*acq['stateVectors']['velocity']))
    
    # Calculate all the satellite positions
    allVals = [(rd['acquisition']['startTime'], rd['acquisition']['numAzimuthSamples']) for rd in radar]
    uniqueTimeSamples = list(set(allVals))
    rdindexes = [[k for k in np.arange(len(radar)) if allVals[k]==uTS] for uTS in uniqueTimeSamples]
    for idxs in rdindexes:
        print("Computing satellite positions for channel %d" % idxs[0])
        #satTimes,satSV = satellitePositionsTime(radar[idxs[0]])
        satTimes,satSV,s,slow = satellitePositionsArclength(radar[idxs[0]])
        #satTimes,satSV = satellitePositions(radar[idxs[0]])
        print(idxs)
        for idx in idxs:
            radar[idx]['acquisition']['satellitePositions'] = satTimes,satSV
            radar[idx]['acquisition']['satelliteArc'] = s,slow
        print("[Done]")
    return radar

#%% Get a reference ground point for simulation
def computeReferenceGroundPoint(radar, radarIDX=None, rTargetIndex=None, sTargetIndex=None):
    if rTargetIndex is None:
        rTargetIndex = 400
    rngs = [(ref['acquisition']['nearRangeTime'] + rTargetIndex*ref['acquisition']['rangeSampleSpacing'])*physical.c/2.0 for ref in radar]
    rng = np.min(rngs)
    if radarIDX is None:
        radarIDX = np.argmin(rngs)
        print("Reference radar object index: %d" % radarIDX)
    #ref = radar[radarIDX]['acquisition']
    
    refTimePos = radar[radarIDX]['acquisition']['satellitePositions']
    rfIDX = int(len(refTimePos[0])/2)
    if sTargetIndex is not None:
        rfIDX = sTargetIndex
        
    satG = sG()
    groundXYZ, error = satG.computeECEF(refTimePos[1][rfIDX,:], 0.0, rng, 0.0)
    print("Ground point computed at time:")
    print(refTimePos[0][rfIDX])
    return groundXYZ, refTimePos[1][rfIDX], refTimePos[0][rfIDX]
    

#%% Function to read acquisition parameters
def readPlatformParametersElement(xmlroot):    
    # Get an instance of the converter
    sconv = stringConverters()
    
    platform = {'lookDirection': sconv.toString('.//lookDirection', xmlroot),
                'orbit': {'inclination': sconv.toAngle('.//inclination', xmlroot), 
                          'orbitAngle': sconv.toAngle('.//orbitAngle', xmlroot),
                          'eccentricity': sconv.toDouble('.//eccentricity', xmlroot),
                          'semiMajorAxis': sconv.toDouble('.//semiMajorAxis', xmlroot),
                          'angleOfPerigee': sconv.toAngle('.//angleOfPerigee', xmlroot),
                          'svec': [sconv.toDouble('.//stateVector/x', xmlroot),
                                   sconv.toDouble('.//stateVector/y', xmlroot),
                                   sconv.toDouble('.//stateVector/z', xmlroot),
                                   sconv.toDouble('.//stateVector/vx', xmlroot),
                                   sconv.toDouble('.//stateVector/vy', xmlroot),
                                   sconv.toDouble('.//stateVector/vz', xmlroot)]},
                'longitude': sconv.toAngle('.//platformLongitude', xmlroot)}
    antenna = {'fc': sconv.toFrequency('.//carrierFrequency', xmlroot),
               'wavelength': physical.c/sconv.toFrequency('.//carrierFrequency', xmlroot),
               'azimuthPositions': sconv.toDistanceArray('.//azimuthPositions', xmlroot),
               'azimuthLengths': sconv.toDistanceArray('.//azimuthElementLengths', xmlroot),
               'transmitPowers': sconv.toPowerArray('.//transmitPowers', xmlroot),
               'elevationPositions': sconv.toDistanceArray('.//elevationPositions', xmlroot),
               'elevationLengths': sconv.toDistanceArray('.//elevationElementLengths', xmlroot),
               'systemLosses': sconv.toPower('.//systemLosses', xmlroot),
               'systemTemperature': sconv.toDouble('.//systemTemperature', xmlroot)
               }
    
    return platform, antenna

#%% Function to read the signal data element
def readSignalDataElement(xmlroot, antenna):
    # Get an instance of the converter
    sconv = stringConverters()
    
    tx = sconv.getXMLElement('.//transmitConfiguration', xmlroot)
    txCols = np.array([m*np.complex(np.cos(p), np.sin(p)) for 
              m,p in zip(sconv.toPowerArray('.//magnitude', tx), sconv.toAngleArray('.//phase', tx))])
    txuZero = sconv.toAngle('.//u0', tx)
    rx = sconv.getXMLElement('.//receiveConfiguration', xmlroot)
    rxCols = np.array([m*np.complex(np.cos(p), np.sin(p)) for 
              m,p in zip(sconv.toPowerArray('.//magnitude', rx), sconv.toAngleArray('.//phase', rx))])
    rxuZero = sconv.toAngle('.//u0', rx)
    
    txMag = np.array(sconv.toPowerArray('.//magnitude', tx))
    txDelay = np.array(sconv.toDurationArray('.//truedelay', tx))
    rxMag = np.array(sconv.toPowerArray('.//magnitude', rx))
    rxDelay = np.array(sconv.toDurationArray('.//truedelay', rx))
    rdrfilename = sconv.toString('.//filename', xmlroot)
    radar = {'acquisition': {'startTime': sconv.toDateTime('.//azimuthStartTime', xmlroot),
                             'numAzimuthSamples': sconv.toInt('.//numAzimuthSamples', xmlroot),
                             'numRangeSamples': sconv.toInt('.//numRangeSamples', xmlroot),
                             'prf': sconv.toDouble('.//pulseRepetitionFrequency', xmlroot),
                             'nearRangeTime': sconv.toDuration('.//nearRangeTime', xmlroot),
                             'rangeSampleSpacing': sconv.toDuration('.//rangeSampleSpacing', xmlroot)},
             'chirp': {'length': sconv.toDuration('.//pulseDuration', xmlroot),
                       'pulseBandwidth': sconv.toFrequency('.//pulseBandwidth', xmlroot)},
             'mode': {'txColumns': txCols, 
                      'rxColumns': rxCols,
                      'txMagnitude': txMag,
                      'txDelay': txDelay,
                      'txuZero': txuZero,
                      'rxuZero': rxuZero,
                      'rxMagnitude': rxMag,
                      'rxDelay': rxDelay},
             'delay': {'baseline': (sum(np.abs(txCols)*antenna['azimuthPositions'])/sum(abs(txCols))/2.0 
                          + sum(np.abs(rxCols)*antenna['azimuthPositions'])/sum(abs(rxCols))/2.0)},
             'filename': rdrfilename
             }
    return radar
        
#%% The state vector function    
def orbit2state(orbit, longitude, svTime):
    if None not in orbit['svec']:
        # State vector already supplied
        sv = state_vector()
        sv.add(svTime, np.array(orbit['svec']))
        return sv
    
    # Calculate a state vector from orbital parameters
    a = orbit['semiMajorAxis']
    e = orbit['eccentricity']
    w = orbit['orbitAngle']
    ia = orbit['inclination']
    l = longitude
    p = orbit['angleOfPerigee']
    P = orbit['period']
    b = a*np.sqrt(1-e**2)
    f = a*e

    
    # Compute the Earth rotation rate
    wE = 2*np.pi/86164.098903691
    
    # Compute the initial position
    s0 = np.array([a*np.cos(w)-f,b*np.sin(w),0])
    
    # Compute the derivatives of the position vector
    ds0 = np.array([-a*np.sin(w),b*np.cos(w),0])
    dds0 = np.array([-a*np.cos(w),-b*np.sin(w),0])
    
    # Compute the transformation matrices
    # ==========================================
    # Compute the matrix to rotate by the argument of perigee
    pM = np.array([[np.cos(p), -np.sin(p), 0],
                   [np.sin(p),  np.cos(p), 0],
                   [0, 0, 1]])
    
    # Compute the matrix to incline the orbit plane
    iM = np.array([[1,       0,        0],
                   [0, np.cos(ia), -np.sin(ia)],
                   [0, np.sin(ia), np.cos(ia)]])
    
    # Compute the matrix to rotate into the ECEF reference frame
    eM = np.array([[np.cos(l),  np.sin(l), 0],
                   [np.sin(l), -np.cos(l), 0],
                   [0,       0, 1]])
    
    # Compute the derivatives of the ECEF rotation matrix
    deM = np.array([[-np.sin(l), np.cos(l), 0],
                    [np.cos(l), np.sin(l), 0],
                    [0,      0, 0]])
             
    ddeM = np.array([[-np.cos(l), -np.sin(l), 0],
                     [-np.sin(l),  np.cos(l), 0],
                     [0,       0, 0]])        
             
    # Compute the ECEF coordinates
    sE = eM.dot(iM).dot(pM).dot(s0)
    
    # Compute the angular velocity from Kepler's second law
    dw = 2*np.pi*a*b/P/((a*np.cos(w)-f)**2 + (b*np.sin(w))**2)
    ddw = (-2*np.pi*a*b/P/((a*np.cos(w)-f)**2 + 
                           (b*np.sin(w))**2)**2*dw*(-2*a*(a*np.cos(w)-f)*np.sin(w)+
                           2*b*b*np.sin(w)*np.cos(w)))
    
    # Compute the ECEF velocity
    dSE = wE*deM.dot(iM).dot(pM).dot(s0) + dw*eM.dot(iM).dot(pM).dot(ds0)
    
    # Compute the ECEF acceleration
    ddSE = (wE**2*ddeM.dot(iM).dot(pM).dot(s0) 
            + 2*dw*wE*deM.dot(iM).dot(pM).dot(ds0)
            + dw**2*eM.dot(iM).dot(pM).dot(dds0) 
            + ddw*eM.dot(iM).dot(pM).dot(ds0))
    
    # Populate the state vector
    sv = state_vector()
    sv.add(svTime, np.append(sE, dSE))
    return sv

#%% Compute state vectors
def satellitePositions(rd):
    nSamples = rd['acquisition']['numAzimuthSamples']
    deltaT = 1.0/rd['acquisition']['prf']
    sampleTimes = [rd['acquisition']['startTime'] + dt.timedelta(seconds=k*deltaT) for k in np.arange(nSamples)]
    svc = rd['platform']['stateVectors'].estimateTimeRange(sampleTimes)
    for tm,sv in zip(sampleTimes, svc):
        rd['platform']['stateVectors'].add(tm, sv)
    return sampleTimes, np.array(svc)

#%% Compute state vectors
def satellitePositionsTime(rd):
    nSamples = rd['acquisition']['numAzimuthSamples']
    deltaT = 1.0/rd['acquisition']['prf']
    sampleTimes = [rd['acquisition']['startTime'] + dt.timedelta(seconds=k*deltaT) for k in np.arange(nSamples)]
    svc = rd['platform']['stateVectors'].estimateTimeRange(sampleTimes)
    for tm,sv in zip(sampleTimes, svc):
        rd['platform']['stateVectors'].add(tm, sv)
    return sampleTimes, np.array(svc)

#%% Compute state vectors
def satellitePositionsArclength(rd):
    nSamples = rd['acquisition']['numAzimuthSamples']
    deltaT = 1.0/rd['acquisition']['prf']

    myrd = rd['platform']['stateVectors']
    xState = myrd.expandedState(myrd.measurementData[0], 0.0)
    reference_time = myrd.measurementTime[0]
    
    ref = rd['acquisition']
    np_prf = np.timedelta64(int(np.round(1e9/ref['prf'])),'ns')
    svTime = ref['startTime'] + np_prf*ref['numAzimuthSamples']/2.0
    print("Reference time:")
    print(reference_time)
    
    #svTime = ref['startTime'] + dt.timedelta(seconds = ref['numAzimuthSamples']/ref['prf']/2.0)
    #sv = orbit2state(rd['platform']['orbit'], rd['platform']['longitude'], svTime)
#        
    
    # Create a slow time object
    C = slow([reference_time])

    # Generate the differntial geometry parameters
    cdf, tdf, T, N, B, kappa, tau, dkappa = C.diffG(xState)

    # Convert time to arclength in the slow-time object
    C.t2s()

    # Get the arclength corresponding to one PRI
    dS = C.ds(deltaT)

    # Get the arclength relative to the state vector time
    start_seconds = (rd['acquisition']['startTime'] - reference_time)/np.timedelta64(1,'s')
    s0 = C.ds(start_seconds)

    sampleS = [s0 + dS*k for k in np.arange(nSamples)]

    def closestRoot(root, val):
        return root[np.argmin(np.abs(root - val))]

    newTimes = np.zeros((nSamples), dtype=float)
    myval = start_seconds
    for k,s in zip(range(nSamples), sampleS):
        root = np.roots([tdf[3]/6.0, tdf[2]/2.0, tdf[1],tdf[0]-s])
        myval = closestRoot(root, myval)
        newTimes[k] = myval

    svc = myrd.estimateTimeRange(myrd.measurementTime,
                                 integrationTimes=newTimes)
    
    def secondsToDelta(s):
        sign = int(np.sign(s))
        ds = np.abs(s)
        secs = int(ds)
        ds = 1e9*(ds - secs)
        nsecs = int(round(ds))
        secs = sign*secs
        nsecs = sign*nsecs
        return np.timedelta64(secs,'s') + np.timedelta64(nsecs,'ns')
        

    tList = newTimes.tolist()
    sampleTimes = [myrd.measurementTime[0]
                   + secondsToDelta(t) for t in tList]
    return sampleTimes, np.array(svc), sampleS, C

#%% plot the space time diagram
def plotSpaceTime(N, radar):
    va = radar[0]['platform']['satelliteVelocity']
    number_CHANNELS = len(radar)
    pos = np.zeros((number_CHANNELS, N))
    tme = np.zeros((number_CHANNELS, N))
    
    leg_string = []
    for rd,k in zip(radar, range(len(radar))):
        t = np.arange(N)/rd['acquisition']['prf']
        tme[k,:] = t+rd['delay']['sampling']
        pos[k,:] = rd['delay']['baseline'] + (t+rd['delay']['sampling'])*va
        leg_string.append('channel %d: PC = %0.3f (m)' % (k,rd['delay']['baseline']))
        leg_string.append('channel %d: x-position' % k)
        
    symbol_ARRAY = ['x','o','+','d','^','v','.','p','h','>','<','s','d','*']
    colors_ARRAY = [(206.0/256, 113.0/256, 200.0/256),
                    (222.0/256, 128.0/256, 91.0/256),
                    (184.0/256, 54.0/256, 50.0/256),
                    (183.0/256, 207.0/256, 105.0/256),
                    (216.0/256, 237.0/256, 217.0/256),
                    (222.0/256, 128.0/256, 91.0/256),
                    (184.0/256, 54.0/256, 50.0/256),
                    (183.0/256, 247.0/256, 105.0/256),
                    (216.0/256, 37.0/256, 217.0/256),
                    (222.0/256, 18.0/256, 91.0/256),
                    (184.0/256, 54.0/256, 50.0/256),
                    (183.0/256, 207.0/256, 105.0/256),
                    (16.0/256, 237.0/256, 217.0/256),
                    (206.0/256, 213.0/256, 59.0/256)]
    
    plt.figure(0)
    sN = len(symbol_ARRAY)
    cN = len(colors_ARRAY)
    for k in range(number_CHANNELS):
        plt.plot(pos[k,:], tme[k,:], symbol_ARRAY[k%sN], color=colors_ARRAY[k%cN])
        plt.plot(pos[k,:], np.zeros(tme[k].shape), symbol_ARRAY[k%sN], color=colors_ARRAY[k%cN])
    plt.grid(True)
    plt.xlabel('Sample position in space (m)')
    plt.ylabel('Sample time (s)')
    plt.show()    

#%% compute the matrix for array factor computation
def arrayFactorMatrix(angles, antenna):
    wavelength = antenna['wavelength']
    beta = 2.0*np.pi/wavelength
    mAngles = np.tile(angles.flatten(), (len(antenna['azimuthPositions']), 1)).T
    position = np.tile(antenna['azimuthPositions'], (len(angles.flatten()), 1))
    return np.exp(-1j*beta*mAngles*position)

#%% Pulse chirp function (One example of a pulse)
class waveform:
    def __init__(self, bandwidth, duration):
        self.bandwidth = bandwidth # double Hz
        self.duration = duration # double s
    
    # Return a function value at a particular time
    def sample(self, t, carrier=0.0):
        pass
    
class chirpWaveform(waveform):
    def sample(self, t, carrier=0.0):
        rate = 2.0*np.pi*np.complex(0,0.5)*self.bandwidth/self.duration
        wcarrier = 2.0*np.pi*np.complex(0,1.0)*carrier
        idx = (t>=0.0) & (t<self.duration)
        y = np.zeros(t.shape, dtype='complex64')
        if carrier == 0.0:
            y[idx] = np.exp( rate*(t[idx]-self.duration/2.0)**2 )
        else:
            y[idx] = np.exp( wcarrier*(t[idx]) + rate*(t[idx]-self.duration/2.0)**2 )
        return y
    
#%% Generate the measured signal
def generateTimeDomainSignal(rad, pointXYZ):
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
                    - 2.0/physical.c*npmat.repmat(rangeMagnitudes,len(fastTime),1))
    chirp = chirpWaveform(rad['chirp']['pulseBandwidth'], rad['chirp']['length'])
    signal = chirp.sample(allFastTimes, rad['antenna']['fc'])*npmat.repmat(antennaWeight,len(fastTime),1)
    return signal
        
#%% Array pattern function (FUNDAMENTAL)
def twoWayArrayPattern(angles, rd, aFM = None):
    anglesShape = angles.shape
    wavelength = rd['antenna']['wavelength']
    if aFM is None:
        aFM = arrayFactorMatrix(angles, rd['antenna'])
    tx_array_factor = np.dot(aFM, rd['mode']['txColumns'])    
    rx_array_factor = np.dot(aFM, rd['mode']['rxColumns'])/np.linalg.norm(rd['mode']['rxColumns'])
    
    minAntennaLength = min(rd['antenna']['azimuthLengths'])
    element_factor = np.sinc(minAntennaLength/wavelength*angles.flatten())**2
    pattern = tx_array_factor*rx_array_factor*element_factor
    return np.reshape(pattern, anglesShape)

#%% Define the function for computing the antenna pattern
def twoWayArrayPatternLinearKS(ks, rad, krz, s_off):
    kShape = ks.shape
    
    # Find the minimum antenna length to use in the element factor
    minAntennaLength = min(rad['antenna']['azimuthLengths'])
    
    # Calculate the antenna element spacing. Assuming here a uniform array
    d = np.mean(np.diff(rad['antenna']['azimuthPositions']))
    
    # Compute the element factor
    eF = np.sinc(-minAntennaLength*ks/4.0/np.pi)**2
    
    # Compute the array factor
    tx = rad['mode']['txMagnitude']
    rx = rad['mode']['rxMagnitude']
    
    # Read the look direction of the channel. It is assumed that all elements
    # are steered by true time delay in the same direction
    txuZero = rad['mode']['txuZero']
    rxuZero = rad['mode']['rxuZero']
    
    # Calculate the polynomial matrix arguments for transmit and receive
    tx1idx = np.where(tx>0.0)
    (tx1len,) = tx1idx[0].shape
    Z = np.exp(1j*d/2.0*(ks + krz*txuZero))
    Ztx = (np.exp(1j*d/2.0*ks)**tx1idx[0][0])*np.exp(-1j*d/4.0*krz*txuZero*(tx1len-1))
    Zd = np.zeros(Z.shape, Z.dtype)
    for k in range(tx1len):
        Zd += Z**k
    Ztx*=Zd

    rx1idx = np.where(rx>0.0)
    (rx1len,) = rx1idx[0].shape
    Z = np.exp(1j*d/2.0*(ks + krz*rxuZero))
    Zrx = (np.exp(1j*d/2.0*ks)**rx1idx[0][0])*np.exp(-1j*d/4.0*krz*rxuZero*(rx1len-1))
    Zd = np.zeros(Z.shape, Z.dtype)
    for k in range(rx1len):
        Zd += Z**k
    Zrx*=Zd
    
    return np.reshape(eF*Ztx*Zrx*np.exp(1j*ks*s_off), kShape)


#%% Compute the signals
def computeSignal(radar, pointXYZ, satSV, rblock_size=None): 
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
        r2t = 2.0/c
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
        n_iterations = len(ranges)
        for pulseIDX in tqdm(range(n_iterations)):
            computeFastTimes = nWCS(fastTimes,
                                    rangeTimes[pulseIDX],
                                    rad['chirp']['length'])
            if computeFastTimes is None:
                continue
            response = np.zeros(len(computeFastTimes), np.complex128)
            antennaResp(response,
                        computeFastTimes, 
                        rangeTimes[pulseIDX],
                        lookDirections[pulseIDX],
                        np.min(rad['antenna']['azimuthLengths']),
                        rad['antenna']['azimuthPositions']/c,
                        rad['mode']['txMagnitude'],
                        rad['mode']['rxMagnitude'],
                        rad['mode']['txDelay'],
                        rad['mode']['rxDelay'],
                        rad['chirp']['pulseBandwidth'],
                        rad['chirp']['length'],
                        rad['antenna']['fc'])
            pulse_data[:,pulseIDX] = response[0:len(fastTimes)]*radarEq1
        
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
                                      rblock = rblock_size)
    return filenames

#%% Define a class to hold sample radar data
class radar_system:
    def __init__(self, radar, bands = None):
        self.radar = radar
        self.n_channels = len(radar)
        rad = radar[int(len(radar)/2)]
        self.f0 = rad['antenna']['fc']
        ref = rad['acquisition']
        self.fs = 1.0/ref['rangeSampleSpacing']
        self.Nr = ref['numRangeSamples']
        self.Na = ref['numAzimuthSamples']
        self.prf = ref['prf']
        self.nearRangeTime = rad['acquisition']['nearRangeTime']
        self.r = (self.nearRangeTime + 
                  np.arange(self.Nr, dtype=np.double)/self.fs)*c/2.0
        self.near_range = self.nearRangeTime*c/2.0
        self.mid_range = (self.nearRangeTime + (self.Nr/2-1)/self.fs)*c/2.0
        self.far_range = (self.nearRangeTime + (self.Nr-1)/self.fs)*c/2.0
        center_sat_index = int(self.Na/2)
        sv = state_vector()
        xState = sv.expandedState(ref['satellitePositions'][1][center_sat_index], 0.0)
        self.expansion_time = ref['satellitePositions'][0][center_sat_index]
        self.sat_position = xState[0,:]
        self.sat_velocity = xState[1,:]
        self.vs = np.linalg.norm(self.sat_velocity)
        
        # Create a slow time object
        self.C = slow(ref['satellitePositions'][0])
        
        # Generate the differential geometry parameters
        cdf, tdf, T, N, B, kappa, tau, dkappa = self.C.diffG(xState)
        
        # Convert time to arclength in the slow-time object
        self.C.t2s()
        
        # Generate the fast time wavenumber
        # Note that since the chirp was mixed to based band, we have the 
        # following expression for the fast time frequency rather than
        # self.kr = (4.0*np.pi/c)*FFT_freq(self.Nr, self.fs, self.f0)
        self.kr = 4.0*np.pi/c*(FFT_freq(self.Nr, self.fs, 0.0) + self.f0)
        self.kridx = np.argsort(self.kr)
        self.kridx_inv = np.argsort(self.kridx)
        self.kr_sorted = self.kr[self.kridx]
        
        # Set up the ks wavenmumber array
        self.ksp = 1.0/(self.C.ds(1.0/self.prf))
        if bands is not None:
            self.update_ks(bands)
            
        # ---------------------------------------------------------------
        # Temporary fix. Fix datatime objects to use numpy.datetime64!
        sqrt_nbands = np.round(np.sqrt(len(radar))).astype('int')
        candidate_offsets = [self.C.ds(1.0/self.prf*k/sqrt_nbands) for k in range(sqrt_nbands)]
        
        # Calculate the sampling offsets for each channel
        s_offsets = [self.C.ds((r['acquisition']['satellitePositions'][0][0] 
              - radar[0]['acquisition']['satellitePositions'][0][0])/np.timedelta64(1,'s')) for r in radar]
        #self.s_offsets = [candidate_offsets[np.argmin(np.abs(soff - candidate_offsets))] for soff in s_offsets]
        self.s_offsets = s_offsets
        # End of temporary fix. Work to do!
        # ---------------------------------------------------------------
        
    def update_ks(self, bands):
        self.bands = bands
        self.n_bands = len(bands)
        self.ks = np.array([2.0*np.pi*FFT_freq(self.Na, self.ksp, b*self.ksp) for b in bands])
        self.ksidx = np.argsort(self.ks.flatten())
        self.ks_full = 2.0*np.pi*FFT_freq(self.Na*self.n_bands, self.ksp*self.n_bands, 0)
        self.ks_full_idx = np.argsort(self.ks_full) - (2 - len(self.ks_full)%2)
        
    def computeGroundPoint(self, radar, radar_idx=None, range_idx=None, azimuth_idx=None):
        pointXYZ, satSV, refTime = computeReferenceGroundPoint(radar, 
                                                      radar_idx,
                                                      range_idx,
                                                      azimuth_idx)
        self.target_ground_position = pointXYZ
        self.target_satellite_position = satSV[0:3]
        self.target_satellite_velocity = satSV[3:]
        self.target_time = refTime
        dt = (self.target_time - self.expansion_time)/np.timedelta64(1,'s')
        self.sx = self.C.ds(dt)
        self.C.computeRangeCoefficients(pointXYZ - satSV[0:3])
        
    def computePhaseResidual(self):
        C = self.radar[0]['acquisition']['satelliteArc'][1]
        s = np.array(self.radar[0]['acquisition']['satelliteArc'][0])
        
        # Get the satpositions from integration and from expansion
        sPos_numerical = self.radar[0]['acquisition']['satellitePositions'][1][:,0:3].T
        sPos_expanded = (np.outer(C.cdf[0], np.ones_like(s)) +
                         np.outer(C.cdf[1], s) +
                         np.outer(C.cdf[2], s**2)/2.0 +
                         np.outer(C.cdf[3], s**3)/6.0)
        sDer_expanded = (np.outer(C.cdf[1], np.ones_like(s)) +
                         np.outer(C.cdf[2], s) +
                         np.outer(C.cdf[3], s**2)/2.0)
        sPos_delta = sPos_numerical - sPos_expanded
    
        # Calculate the domain variable
        Rn_vector = sPos_numerical - np.outer(self.target_ground_position, 
                                              np.ones_like(s))
        Rn = np.linalg.norm(Rn_vector, axis=0)
        rhat_vector = Rn_vector/np.outer(np.ones((3,)), Rn)
        ksM = -self.kr[0]*np.sum(rhat_vector*sDer_expanded, axis=0)
    
        # Caluclate component of delta in direction of rhat (dependent variable)
        delta = -self.kr[0]*np.sum(sPos_delta*rhat_vector, axis=0)
        
        # Compute the interpolator
        f = interpolate.interp1d(ksM, delta, kind = 'linear')
        
        # Get the ks dependent values
        self.ks_phase_correction = f(self.ks_full)
        

#%% Compute and store the r_sys file for quicker processing
def computeStoreRsys(radar, bands = None, ref_range_idx = None):
    sys_file = fio.fileStruct(radar[0]['filename'], "", "", "rsys.pickle")
    
    """Define bands if not defined"""
    if bands is None:
        bands = np.arange(-int(len(radar)/2)-2,int(len(radar)/2)+1+2)
    
    print("Computing radar parameters...")
    r_sys = radar_system(radar, bands)
    
    print("Sample offsets")
    print(r_sys.s_offsets)
        
    if ref_range_idx is None:
        ref_range_idx = int(radar[0]['acquisition']['numRangeSamples']/2)
    print("Computing arclength parameters at reference range index %d..." % ref_range_idx)
    r_sys.computeGroundPoint(radar, range_idx=ref_range_idx)
    
    with open(sys_file, 'wb') as f:
        pickle.dump(r_sys, f, pickle.HIGHEST_PROTOCOL)
        
    return r_sys
        
#%% Compute the processing filter or load from file
def computeStoreMultiProcFilter(radar):
    mprocH_file = fio.fileStruct(radar[0]['filename'], "", "", "mprocH.pickle")
    
    r_sys = loadRsys(radar)
    
    print("Computing the multi-channel H matrix...")
    H = np.array([twoWayArrayPatternLinearKS(r_sys.ks, 
                                             rd, 
                                             r_sys.kr[0], 
                                             ds) 
                  for rd,ds in zip(radar,r_sys.s_offsets)])
        
    """Normalize the computed pattern"""
    H /= np.sqrt(np.max(H*np.conj(H)))
    
    print("Writing processing filter to pickle file...")
    with open(mprocH_file, 'wb') as f:
        pickle.dump(H, f, pickle.HIGHEST_PROTOCOL)
    
    """Return the processing filter"""
    return H, r_sys

#%% Compute the processing filter or load from file
def loadMultiProcFilter(radar):
    mprocH_file = fio.fileStruct(radar[0]['filename'], "", "", "mprocH.pickle")
    print("Loading processing filter from pickle file...")
    print(mprocH_file)
    try:
        with open(mprocH_file, 'rb') as f:
           H = pickle.load(f)
    except FileNotFoundError as fE:
        print(fE)
        print("No H file loaded, please calculate and store")
        H = None
       
    """Return the processing filter"""
    return H

#%% Compute the processing filter or load from file
def loadRsys(radar):
    sys_file = fio.fileStruct(radar[0]['filename'], "", "", "rsys.pickle")
    print("Load r_sys from pickle file...")
    print(sys_file)
    try:
        with open(sys_file, 'rb') as f:
           r_sys = pickle.load(f)
    except FileNotFoundError as fE:
        print(fE)
        print("No r_sys file loaded, please calculate and store")
        r_sys = None
    
    """Return the processing filter"""
    return r_sys

#%% Compute the inverse matrices
def multiChannelProcessMem(radar,
                           rangeidx=[0,-1],
                           p=0.5, 
                           SNR = 1.0,
                           make_plots=False):
    
    # print("Get the multi-channel H matrix...")
    r_sys = loadRsys(radar)
    H = loadMultiProcFilter(radar)
    
    # print("Compute the desired antenna pattern...")
    D = np.sqrt(np.sum(np.abs(H)**2, axis=0))
    if make_plots:
        plt.figure()
        plt.plot(sorted(r_sys.ks_full), D.flatten()[r_sys.ksidx])
        plt.title("Desired antenna pattern")
        plt.xlabel("Azimuth wavenumber (m$^{-1}$)")
        plt.ylabel("Gain (Natural units)")
        plt.grid()
        plt.show()
        
    # print("Loading the data from disk...")
    data = fio.loadNumpy_raw_dataMem(radar, 
                                     ridx = rangeidx)
    _, n_r, n_x = data.shape
    n_b = r_sys.n_bands
    
    """Defining the noise covariance"""
    Rn = np.eye(len(radar))/np.sqrt(SNR)
    
    # print("Multi-channel processing ...")
    procData = np.zeros((n_b,n_x,n_r), dtype=np.complex128)
    for kidx in tqdm(range(n_x)):
        Rinv = np.linalg.inv(np.dot(H[:,:,kidx], np.conj(H[:,:,kidx].T)) 
                                    + (1.0-p)/p*Rn)
        B = np.dot(np.diag(D[:,kidx]), np.dot(np.conj(H[:,:,kidx].T), Rinv))
        if np.any(np.isnan(B)):
            print(kidx)
            break
        dummy = np.dot(B,data[:,:,kidx])
        for bidx in range(n_b):
            procData[bidx,kidx,:] = dummy[bidx,:]
    
    # Release data from memory
    del data
            
    # Reorder the data
    # print("Re-ordering data and returning...")
    procData = procData.reshape((n_b*n_x, n_r))
    for k in tqdm(np.arange(n_r)):
        procData[:,k] = procData[r_sys.ksidx[r_sys.ks_full_idx], k]
    return procData, r_sys
   
#%% Compute the inverse matrices
def multiChannelProcess(radar, bands=None, p=0.5, make_plots=False, SNR = 1.0):
    """Define bands if not defined"""
    if bands is None:
        bands = np.arange(-int(len(radar)/2)-2,int(len(radar)/2)+1+2)
    
    print("Loading the data from disk...")
    data = loadNumpy_raw_data(radar)
    
    print("Computing radar parameters...")
    r_sys = radar_system(radar, bands)
    print("Sample offsets")
    print(r_sys.s_offsets)
    
    print("Computing the multi-channel H matrix...")
    H = computeStoreMultiProcFilter(radar, bands=None)
    H = np.array([twoWayArrayPatternLinearKS(r_sys.ks, rd, r_sys.kr[0], ds) 
                  for rd,ds in zip(radar,r_sys.s_offsets)])
        
    """Normalize the computed pattern"""
    H /= np.sqrt(np.max(H*np.conj(H)))
    
    print("Computing the desired antenna pattern...")
    D = np.sqrt(np.sum(np.abs(H)**2, axis=0))
    if make_plots:
        plt.figure()
        plt.plot(sorted(r_sys.ks_full), D.flatten()[r_sys.ksidx])
        plt.title("Desired antenna pattern")
        plt.xlabel("Azimuth wavenumber (m$^{-1}$)")
        plt.ylabel("Gain (Natural units)")
        plt.grid()
        plt.show()
    
    """Defining the noise covariance"""
    Rn = np.eye(len(radar))/np.sqrt(SNR)
    
    print("Multi-channel processing ...")
    procData = np.zeros((r_sys.n_bands,r_sys.Na,r_sys.Nr), dtype=np.complex128)
    blkSize = int(r_sys.Na/20)
    for kidx in range(r_sys.Na):
        if kidx%blkSize == 1:
            print("Progress: %0.2f" % (100.0*kidx/r_sys.Na))
        Rinv = np.linalg.inv(np.dot(H[:,:,kidx], np.conj(H[:,:,kidx].T)) 
                                    + (1.0-p)/p*Rn)
        B = np.dot(np.diag(D[:,kidx]), np.dot(np.conj(H[:,:,kidx].T), Rinv))
        if np.any(np.isnan(B)):
            print(kidx)
            break
        dummy = np.dot(B,data[:,:,kidx])
        for bidx in range(r_sys.n_bands):
            procData[bidx,kidx,:] = dummy[bidx,:]
    
    # Release data from memory
    del data
            
    # Reorder the data
    print("Re-ordering data and returning...")
    procData = procData.reshape((r_sys.n_bands*r_sys.Na, r_sys.Nr))
    blkSize = int(r_sys.Nr/20)
    for k in np.arange(r_sys.Nr):
        if k%blkSize ==1:
            print("Progress: %0.2f" % (100.0*k/r_sys.Nr))
        procData[:,k] = procData[r_sys.ksidx[r_sys.ks_full_idx], k]
    return procData, r_sys

#%% SAR process the signal. Many assumptions are made about the signal
def wkProcessNumba(procData, 
                   r_sys,
                   ks = None,
                   r = None, 
                   os_factor = 8, 
                   mem_cols = 1024, 
                   mem_rows = 8192, 
                   tempFile = None):
    """ 
    mem_cols are the number of columns to process at one go. The data are
    organised by
            az1 az2 az3 ...
    rng1
    rng2
    rng2
    .
    .
    .
    
    Thus, mem_cols defines the number of azimuth samples to process in one 
    call
    """
    
    try:
        r = r or np.linalg.norm(r_sys.C.R)
    except AttributeError as aE:
        print("will use center range for w-k processing")
        r = r_sys.mid_range
        
    """ Define the ks frequencies if not already defined """
    if ks is None:
        ks = r_sys.ks_full
        
    """ Calculate the inverse frequency offset for oversampling in
        the frequency signal (this is done in the interpolation step)"""
    swath = r_sys.far_range - r_sys.near_range
    baseband_mix = 2.0*np.pi*np.mod(r_sys.mid_range, swath)/swath
    # baseband_mix = 2.0*np.pi*0.5/r_sys.Nr
        
    r2t = 2.0/c
    cols = len(r_sys.kr_sorted)
    rows = len(ks)
    
    """ Allocate memory and calculate indeces for the pulse workspace """
    Yos_idx = np.round(FFT_freq(cols,cols,0)).astype('int')
    
    dkr = r_sys.kr[1]-r_sys.kr[0]
    
    row_ticks = list(range(0,rows,mem_rows)) + [rows]
    row_spans = [(row_ticks[k], row_ticks[k+1]) 
                 for k in range(len(row_ticks)-1)]
    
    wk_processed = np.zeros((rows,mem_cols), dtype=np.complex128)
    yupkr = 4.0*np.pi/c*(FFT_freq(cols*os_factor, r_sys.fs, 0) + r_sys.f0)
    yupidx = np.argsort(yupkr)
    
    for span in row_spans:
        # print("Processing rows (azimuth) %d to %d..." % span)
        n_rows = span[1]-span[0]
        YY = np.zeros((n_rows, cols), dtype=np.complex128)
    
        iP = np.zeros((n_rows, r_sys.kr_sorted.shape[0]), 
                      dtype=np.double)
        
        """ Compute the points at which to interpolate """
        stlz.getInterpolationPoints(ks[span[0]:span[1]], 
                                    r_sys.kr_sorted,
                                    iP,
                                    r,
                                    r_sys.C.a2,
                                    r_sys.C.a3,
                                    r_sys.C.a4,
                                    max_iter=10,
                                    error_tol = 1e-10)
        # iP = np.zeros((n_rows, r_sys.kr_sorted.shape[0]), 
        #       dtype=np.double)
        
        # """ Compute the points at which to interpolate """
        # nbwk.getInterpolationPoints(ks[span[0]:span[1]], 
        #                             r_sys.kr_sorted,
        #                             iP,
        #                             r,
        #                             r_sys.C.a2,
        #                             r_sys.C.a3,
        #                             r_sys.C.a4,
        #                             max_iter=10,
        #                             error_tol = 1e-10)
        
        """ Perform the interpolation """
        print("Interpolating the signal...")
        nbwk.interpolatePulsesCxSimple(procData[span[0]:span[1], :], 
                                        YY,
                                        (iP - r_sys.kr_sorted[0])/dkr, 
                                        os_factor, 
                                        yupidx)
        # Yos = np.zeros((cols*os_factor, ), dtype=np.complex128)
        # chunk_DATA = np.fft.fft(procData[span[0]:span[1], :], axis=1)
        # nbwk.interpolatePulsesCx(chunk_DATA[:,r_sys.kridx], 
        #                         YY,
        #                         (iP - r_sys.kr_sorted[0])/dkr, 
        #                         os_factor, 
        #                         Yos_idx,
        #                         baseband_mix)
        
        
        """ Re-order the data"""
        iP = iP[:,r_sys.kridx_inv]
        YY = YY[:,r_sys.kridx_inv]
    
        #% Phase multiply the signal
        print("Correcting residual phase and applying IFFT...")
        KR = np.matlib.repmat(r_sys.kr, n_rows, 1)
        wkSignal = np.fft.ifft(YY*np.exp(-1j*(iP-KR)*r_sys.r[0]), axis=1)
        wk_processed[span[0]:span[1],:] = wkSignal[:,0:mem_cols]
        print("[DONE]")
        
    return wk_processed
    
#%% Define a function to load the data with given file naming
# convention. This loads the raw data generated as measurements from
# the sensor. These data need to be multi-channel processed
def loadNumpy_raw_data(radar, target_domain = "rX"):
    # Process the file names
    fls = [r['filename'] for r in radar]
    domain = [fl.split("_")[-2] for fl in fls]
    print(fls)
    
    # Define the data loading function dictionary
    fn_dict = {'rxrX': lambda x: np.fft.fft(x, axis=1), 
               'RxrX': lambda x: np.fft.ifft(np.fft.fft(x, axis=1), axis=0),
               'rXrX': lambda x: x,
               'RXrX': lambda x: np.fft.ifft(x, axis=0),
               'rxrx': lambda x: x, 
               'Rxrx': lambda x: np.fft.ifft(x, axis=0),
               'rXrx': lambda x: np.fft.ifft(x, axis=1),
               'RXrx': lambda x: np.fft.ifft2(x),
               'rxRX': lambda x: np.fft.fft2(x), 
               'RxRX': lambda x: np.fft.fft(x, axis=1),
               'rXRX': lambda x: np.fft.fft(x, axis=0),
               'RXRX': lambda x: x,
               'rxRx': lambda x: np.fft.fft(x, axis=0), 
               'RxRx': lambda x: x,
               'rXRx': lambda x: np.fft.ifft(np.fft.fft(x, axis=0), axis=1),
               'RXRx': lambda x: np.fft.ifft(x, axis=1)}
    
    # Load the data
    #data = np.stack([np.fft.fft(np.load(fl), axis=1) for fl in fls], axis=0)
    return np.stack([fn_dict[dm + target_domain](np.load(fl)) for dm,fl in zip(domain,fls)], axis=0)


#%% Define a function to load the multichannel processed data. These
# data will need to be processed by a standard "stripmap" SAR processor
# like the Omega-K processor
def loadNumpy_mcp_data(data_file):
    fn_dict = {'rx': lambda x: np.fft.fft2(x, axis=1), 
           'Rx': lambda x: np.fft.fft(x, axis=1),
           'rX': lambda x: np.fft.fft(x, axis=0),
           'RX': lambda x: x,
           'xr': lambda x: np.fft.fft2(x).T, 
           'xR': lambda x: np.fft.fft(x, axis=0).T,
           'Xr': lambda x: np.fft.fft(x, axis=1).T,
           'XR': lambda x: x.T}
    domain = data_file.split("_")[-2]
    return fn_dict[domain](np.load(data_file))
