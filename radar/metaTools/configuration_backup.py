#%% imports and constants
import xml.etree.ElementTree as etree
import numpy as np
import numpy.matlib as npmat
import datetime as dt
import sys
sys.path.append('/home/sikaneta/local/src/Python/radar')
import matplotlib.pyplot as plt
from measurement.measurement import state_vector
from geoComputer.geoComputer import satGeometry as sG


defaultConfig = u'/home/sikaneta/local/Matlab/Delph2014/sureConfig.xml'

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
            return dt.datetime(self.toInt('.//year', targetElement),
                               self.toInt('.//month', targetElement),
                               self.toInt('.//day', targetElement),
                               self.toInt('.//hour', targetElement),
                               self.toInt('.//minute', targetElement),
                               self.toInt('.//sec', targetElement),
                               self.toInt('.//usec', targetElement))
        else:
            return None
        
        
#%% The configuration ode        
def loadConfiguration(configFile = None):
    # Make sure there is a config file
    configFile = configFile or defaultConfig
    
    # Get the root Element
    xmlroot = etree.parse(configFile).getroot()
    
    (platform,antenna) = readPlatformParametersElement(xmlroot.find('.//acquisitionParameters'))
    radar = [readSignalDataElement(sigData, antenna) for sigData in xmlroot.findall('.//signalData')]
    
    refTime = dt.datetime(2000,1,1,0,0,0)
    secs2K = [(rd['acquisition']['startTime']- refTime).total_seconds() for rd in radar]
    
    min2K = min(secs2K)
    sdelay = [s2K-min2K for s2K in secs2K]
    for rd, sd in zip(radar, sdelay):
        rd['delay']['sampling'] = sd
    
    # Calculate the time for the state vector calculation
    ref = radar[0]['acquisition']
    svTime = ref['startTime'] + dt.timedelta(seconds = ref['numAzimuthSamples']/ref['prf']/2.0)
    rng = (ref['nearRangeTime'] + ref['numRangeSamples']*ref['rangeSampleSpacing'])*physical.c/2.0
    
    platform['orbit']['period'] = np.sqrt(4.0*np.pi**2/(physical.G*physical.mass_EARTH)*platform['orbit']['semiMajorAxis']**3)
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
        print "Computing satellite positions for channel %d" % idxs[0]
        satTimes,satSV = satellitePositions(radar[idxs[0]])
        for idx in idxs:
            radar[idx]['acquisition']['satellitePositions'] = satTimes,satSV
        print "[Done]"
    return radar

#%% Get a reference ground ponit for simulation
def computeReferenceGroundPoint(radar, radarIDX=None):
    rngs = [(ref['acquisition']['nearRangeTime'] + ref['acquisition']['numRangeSamples']*ref['acquisition']['rangeSampleSpacing']/10.0)*physical.c/2.0 for ref in radar]
    rng = np.min(rngs)
    radarIDX = radarIDX or np.argmin(rngs)
    ref = radar[radarIDX]['acquisition']
    
    refTimePos = radar[radarIDX]['acquisition']['satellitePositions']
    rfIDX = len(refTimePos[0])/2
    satG = sG()
    groundXYZ, error = satG.computeECEF(refTimePos[1][rfIDX,:], 0.0, rng, 0.0)
    return groundXYZ, refTimePos[1][rfIDX,0:3]
    

#%% Function to read acquisition parameters
def readPlatformParametersElement(xmlroot):    
    # Get an instance of the converter
    sconv = stringConverters()
    
    platform = {'lookDirection': sconv.toString('.//lookDirection', xmlroot),
                'orbit': {'inclination': sconv.toAngle('.//inclination', xmlroot), 
                          'orbitAngle': sconv.toAngle('.//orbitAngle', xmlroot),
                          'eccentricity': sconv.toDouble('.//eccentricity', xmlroot),
                          'semiMajorAxis': sconv.toDouble('.//semiMajorAxis', xmlroot),
                          'angleOfPerigee': sconv.toAngle('.//angleOfPerigee', xmlroot)}, 
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
    
                
#    antenna = {'fc': sconv.toFrequency('.//carrierFrequency', xmlroot),
#               'wavelength': physical.c/sconv.toFrequency('.//carrierFrequency', xmlroot),
#               'azimuthPositions': sconv.toDistanceArray('.//azimuthPositions', xmlroot),
#               'azimuthLengths': sconv.toDistanceArray('.//azimuthElementLengths', xmlroot),
#               'transmitPowers': sconv.toPowerArray('.//transmitPowers', xmlroot),
#               'elevationPositions': sconv.toDistanceArray('.//elevationPositions', xmlroot),
#               'elevationLengths': sconv.toDistanceArray('.//elevationElementLengths', xmlroot),
#               'systemLosses': sconv.toPower('.//systemLosses', xmlroot),
#               'systemTemperature': sconv.toDouble('.//systemTemperature', xmlroot)
#               } 
    

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
                          + sum(np.abs(rxCols)*antenna['azimuthPositions'])/sum(abs(rxCols))/2.0)}
             }
    return radar
        
#%% The state vector function    
def orbit2state(orbit, longitude, svTime):
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

#%% fft freq function
def FFT_freq(N, fp, f0):
    freq = np.arange(float(N))*fp/N
    fidx = (freq/fp>0.5)
    freq[fidx] = freq[fidx] - fp
    freq = freq+np.round(-(freq-f0)/fp)*fp   
    return freq

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
def twoWayArrayPatternTrueTime(fastTime, lookDirections, ranges, rad, Ztx = None, Zrx = None):
    # We have the same number of angles as pulses in this model as the pulse
    # returns from a single point target as the radar travels. The angles correspond
    # to the radar look direction
    
    # Find an effective wavelegnth for the element pattern
    wavelength = rad['antenna']['wavelength']
    
    # Find the minimum antenna length to use in the element factor
    minAntennaLength = min(rad['antenna']['azimuthLengths'])
    
    # Calculate the antenna element spacing. Assuming here a uniform array
    d = np.mean(np.diff(rad['antenna']['azimuthPositions']))
    
    # Compute the element factor
    eF = np.matlib.repmat(np.sinc(minAntennaLength/wavelength*lookDirections.flatten())**2, len(fastTime), 1)
    
    # Compute the fast time frequency values
    w = 2.0*np.pi*FFT_freq(len(fastTime), 1.0/ref['rangeSampleSpacing'], physical.c/wavelength)
    #w = 2.0*np.pi*FFT_freq(len(fastTime), 1.0/ref['rangeSampleSpacing'], 0.0) + 2.0*np.pi*physical.c/wavelength
    W = np.matlib.repmat(w, len(u), 1).T
    
    # Create a matrix of azimuth positions
    X = np.matlib.repmat(rad['antenna']['azimuthPositions'], len(w), 1)
    
    # Read the antenna element weighting values
    tx = rad['mode']['txMagnitude']
    rx = rad['mode']['rxMagnitude']
    
    # Read the look direction of the channel. It is assumed that all elements
    # are steered by true time delay in the same direction
    txuZero = rad['mode']['txuZero']
    rxuZero = rad['mode']['rxuZero']
    
    # Compute the azimuth component of the signal
    Az = np.exp(-1j*2.0/physical.c*np.matlib.repmat(ranges,len(w),1)*W)
    
    # Repeat the look directions across all frequencies
    U = np.matlib.repmat(lookDirections,len(w),1)
    
    # Calculate the polynomial matrix arguments for transmit and receive
    print tx.shape
    print U.shape
    print W.shape
    if Ztx is None:
        Ztx = np.polyval(tx, np.exp(-1j*d/physical.c*W*(U-txuZero)))
    if Zrx is None:
        Zrx = np.polyval(rx, np.exp(-1j*d/physical.c*W*(U-rxuZero)))
    
    # Compute the chirp and tranform into frequency domain
    chirp = chirpWaveform(rad['chirp']['pulseBandwidth'], rad['chirp']['length'])
    S = np.matlib.repmat(np.fft.fft(chirp.sample(fastTime - rad['acquisition']['nearRangeTime'], carrier=rad['antenna']['fc'])), len(ranges),1).T
    D = np.exp(1j*W*rad['acquisition']['nearRangeTime'])
    
    # Calculate the return signal in the fast-time frequency domain
    Z = S*D*Az*eF*Ztx*Zrx
    
    # Return the time domain signal
    return eF*np.fft.ifft(Z, axis=0)/len(fastTime), Ztx, Zrx

#%% Array pattern function (FUNDAMENTAL)
def twoWayArrayPatternTrueTimeFlat(fastTime, lookDirections, ranges, rad, Ztx = None, Zrx = None, FFTLen = 16384):
    # We have the same number of angles as pulses in this model as the pulse
    # returns from a single point target as the radar travels. The angles correspond
    # to the radar look direction
    
    # Find an effective wavelegnth for the element pattern
    wavelength = rad['antenna']['wavelength']
    
    # Find the minimum antenna length to use in the element factor
    minAntennaLength = min(rad['antenna']['azimuthLengths'])
    
    # Calculate the antenna element spacing. Assuming here a uniform array
    d = np.mean(np.diff(rad['antenna']['azimuthPositions']))
    
    # Compute the element factor
    eF = np.matlib.repmat(np.sinc(minAntennaLength/wavelength*lookDirections.flatten())**2, len(fastTime), 1)
    
    # Compute the fast time frequency values
    w = 2.0*np.pi*FFT_freq(len(fastTime), 1.0/ref['rangeSampleSpacing'], physical.c/wavelength)
    #w = 2.0*np.pi*FFT_freq(len(fastTime), 1.0/ref['rangeSampleSpacing'], 0.0) + 2.0*np.pi*physical.c/wavelength
    W = np.matlib.repmat(w, len(u), 1).T
    
    # Create a matrix of azimuth positions
    X = np.matlib.repmat(rad['antenna']['azimuthPositions'], len(w), 1)
    
    # Read the antenna element weighting values
    tx = rad['mode']['txMagnitude']
    rx = rad['mode']['rxMagnitude']
    
    # Read the look direction of the channel. It is assumed that all elements
    # are steered by true time delay in the same direction
    txuZero = rad['mode']['txuZero']
    rxuZero = rad['mode']['rxuZero']
    
    # Compute the azimuth component of the signal
    Az = np.exp(-1j*2.0/physical.c*np.matlib.repmat(ranges,len(w),1)*W)
    
    # Repeat the look directions across all frequencies
    U = np.matlib.repmat(lookDirections,len(w),1)
    
    # Calculate the polynomial matrix arguments for transmit and receive
    if Ztx is None:
        tx1idx = np.where(tx>0.0)
        (tx1len,) = tx1idx[0].shape
        Z = np.exp(-1j*d/physical.c*W*(U-txuZero))
        Ztx = (Z**tx1idx[0][0])*(1.0-Z**tx1len)/(1.0-Z)
        Z = None
    if Zrx is None:
        rx1idx = np.where(rx>0.0)
        (rx1len,) = rx1idx[0].shape
        Z = np.exp(-1j*d/physical.c*W*(U-txuZero))
        Zrx = (Z**rx1idx[0][0])*(1.0-Z**rx1len)/(1.0-Z)
        Z = None
    print tx.shape
    print U.shape
    print W.shape
    
    # Compute the chirp and tranform into frequency domain
    chirp = chirpWaveform(rad['chirp']['pulseBandwidth'], rad['chirp']['length'])
    S = np.matlib.repmat(np.fft.fft(chirp.sample(fastTime - rad['acquisition']['nearRangeTime'], carrier=rad['antenna']['fc'])), len(ranges),1).T
    D = np.exp(1j*W*rad['acquisition']['nearRangeTime'])
    
    # Calculate the return signal in the fast-time frequency domain
    Z = S*D*Az*eF*Ztx*Zrx
    
    # Return the time domain signal
    # return np.fft.fft((eF*np.fft.ifft(Z, axis=0)/len(fastTime)).astype('complex64'), n=16384, axis=0), Ztx, Zrx
    return eF*np.fft.ifft(Z, axis=0)/len(fastTime), Ztx, Zrx

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


#%% generate the multichannel signal
def generateAndFilter(radar, p=0.1):
    platform = radar[0]['platform']
    antenna = radar[0]['antenna']
    bandwidth = platform['satelliteVelocity']/antenna['wavelength']
    pulses = radar[0]['acquisition']['numAzimuthSamples']
    va = platform['satelliteVelocity']
    veff = 7.221125425e3
    ref = radar[0]['acquisition']
    prf = radar[0]['acquisition']['prf']
    wavelength = antenna['wavelength']
    nChan = len(radar)
    rng = (ref['nearRangeTime'] + ref['numRangeSamples']*ref['rangeSampleSpacing']/2.0)*physical.c/2.0
    dAmb = prf*wavelength*rng/2.0/(veff**2/va)
    print "Range: %0.6f" % rng
    print "Wavelength %0.6f" % wavelength
    
    bands = np.arange(-np.floor(np.ceil(bandwidth/prf)/2),np.floor(np.ceil(bandwidth/prf)/2)+1)
    dop_CEN = 0.0
    fd = FFT_freq(pulses, prf, dop_CEN)
    freq = np.zeros((len(fd), len(bands)))
    for k,band in zip(range(len(bands)), bands):
        freq[:,k] = fd + band*prf
    
    chirp = np.exp(-1j*4*np.pi/wavelength*rng*np.sqrt(1-(wavelength*freq/2.0/veff)**2))
    u1 = -(freq)*wavelength/2.0/va
    
    # Calculate the array factor matrix to save computation
    aFM = arrayFactorMatrix(u1, radar[0]['antenna'])
    
    arrSize = (len(radar), pulses, len(bands))
    xsig = np.zeros(arrSize, dtype=np.complex64)
    hsig = np.zeros(arrSize, dtype=np.complex64)
    B = np.zeros((len(fd), len(bands), nChan), dtype=np.complex64)
    
    for k in range(len(radar)):
        if(k%10==1):
            print 'Generating unambiguous signal for channel %d of %d' % (k+1, nChan)
        a1 = twoWayArrayPattern(u1, radar[k], aFM)
        delay = np.exp(1j*2.0*np.pi*freq*radar[k]['delay']['sampling'])
        xsig[k,:,:] = a1*chirp*delay
        hsig[k,:,:] = a1*delay
        
    measuredData = {}
    measuredData['fd'] = freq
    measuredData['chirp'] = chirp
    measuredData['raw'] = {'sampled': np.sum(xsig, 2)}
    measuredData['D'] = np.squeeze(np.sqrt(np.sum(np.abs(hsig)**2, 0)/nChan))
    measuredData['Rn'] = p*np.mean(np.abs(measuredData['raw']['sampled'].flatten()**2))*np.eye(nChan)
    for k in range(len(fd)):
        if(k%100==1):
            print 'Calculating filters for sampled frequency %f' % fd[k]
        H = np.squeeze(hsig[:,k,:])
        B[k,:,:] = np.diag(measuredData['D'][k,:]).dot(H.conj().T).dot(np.linalg.inv(H.dot(H.conj().T)+measuredData['Rn']))
        
    measuredData['B'] = B
    
    # Apply the filters to the backfolded signal
    measuredData['filtered'] = {'signal': np.zeros((len(fd), len(bands)), dtype=np.complex64),
                                'noise': np.zeros((len(fd), len(bands)), dtype=np.complex64)}
    (sigM, sigN) = measuredData['raw']['sampled'].shape
    measuredData['raw']['noise'] = (np.sqrt(np.mean(np.abs(measuredData['raw']['sampled'].flatten())**2)/2)
                            *(np.random.randn(sigM, sigN)+1j*np.random.randn(sigM, sigN)))
    for k in range(len(fd)):
        if(k%100==1):
            print 'Applying filters for sampled frequency %f' % fd[k]
        measuredData['filtered']['signal'][k,:] = np.squeeze(measuredData['B'][k,:,:]).dot(measuredData['raw']['sampled'][:,k])
        measuredData['filtered']['noise'][k,:] = np.squeeze(measuredData['B'][k,:,:]).dot(measuredData['raw']['noise'][:,k])
        
    SNR_preprocessing = np.mean(np.abs(measuredData['raw']['sampled']).flatten()**2)/np.mean(np.abs(measuredData['raw']['noise']).flatten()**2)
    SNR_postprocessing = np.mean(np.abs(measuredData['filtered']['signal']).flatten()**2)/np.mean(np.abs(measuredData['filtered']['noise']).flatten()**2)
    
    print "Pre-processing SNR %0.4f" % (20.0*np.log10(SNR_preprocessing))
    print "Post-processing SNR %0.4f" % (20.0*np.log10(SNR_postprocessing))
    measuredData['raw']['SNR'] = (20.0*np.log10(SNR_preprocessing))
    measuredData['filtered']['SNR'] = (20.0*np.log10(SNR_postprocessing))
    measuredData['ambiguity'] = dAmb
    return measuredData

#%% Analyze the filters
def analyzeFilters(mD, radar):      
    processedData = {'target': mD['filtered']['signal']*np.conj(mD['chirp']),
                     'noise': mD['filtered']['noise']*np.conj(mD['chirp'])}
    fidx = np.argsort(mD['fd'].flatten())
    SAR = {'target': np.fft.fftshift(np.fft.ifft(processedData['target'].flatten()[fidx])),
           'noise': np.fft.fftshift(np.fft.ifft(processedData['noise'].flatten()[fidx]))}
    nrmF = max(abs(SAR['target']))
    SAR['target'] = SAR['target']/nrmF
    SAR['noise'] = SAR['noise']/nrmF
    mP = len(SAR['target'])
    dx = radar[0]['platform']['satelliteVelocity']/(2.0*np.max(mD['fd']))
    sPos = (np.arange(mP) - mP/2.0)*dx
    minPosIdx = np.argmin(np.abs(sPos))
    beamMax = np.argmax(np.abs(SAR['target']))
    SAR['pos'] = sPos -sPos[beamMax]+sPos[minPosIdx]
    processedData['SAR'] = SAR
    plt.figure(1)
    plt.plot(SAR['pos'], 20*np.log10(np.abs(SAR['target'])))
    plt.grid(True)
    plt.xlabel('xPos (m)')
    plt.ylabel('Response (dB)')
    
    plt.figure(2)
    plt.plot(mD['fd'].flatten()[fidx[0::10]], 20.0*np.log10(np.abs(mD['filtered']['signal'].flatten()[fidx[0::10]])))
    plt.grid(True)
    plt.xlabel('Doppler frequency (Hz)')
    plt.ylabel('Reconstructed signal (dB)')
    return processedData