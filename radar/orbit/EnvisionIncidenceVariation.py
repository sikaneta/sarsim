# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 14:22:02 2022

@author: ishuwa.sikaneta
"""

#%%
from orbit.geometry import surfaceNormal as sNorm
from orbit.geometry import findNearest
from orbit.geometry import computeImagingGeometry
from orbit.orientation import orbit
from orbit.envision import loadSV
from sarsimlog import logger

from tqdm import tqdm
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import json
import os
import sys

#%% Define class to capture convergence errors
class ConvergenceError(Exception):
    def __init__(self, algorithm, maxiter):
        self.algorithm = algorithm
        self.maxiter = maxiter
    def __str__(self):
        return '(%s) did not converge after %d iterations' % (self.algorithm, 
                                                              self.maxiter)

#%% Define class to capture convergence errors
class BoundsError(Exception):
    def __init__(self, predicted, lowerBound, upperBound):
        self.index = predicted
        self.lowerBound = lowerBound
        self.upperBound = upperBound
    def __str__(self):
        return 'Value: %d outside of set [%d, %d]' % (self.index, 
                                                      self.lowerBound, 
                                                      self.upperBound)
        
#%%
class envisionIncidence:
    
    def __init__(self, do_plots = False):
        self.plots = do_plots
        self.__tascarr = []
        self.__idxarr = []
        self.__sv = None
        self.__cycle_jump = []
        self.estimateOrbitStartTimes() #%% Reloads state vectors but in VCR frame
        self.__sv = loadSV(toPCR = True)
    
    def getApproximateAscNodeTimes(self):
        return self.__tascarr, self.__idxarr
    
    def estimateOrbitAngleTimes(self, targetOrbitAngle):
        """
        Scan the state vector file for times where the satellite position
        is at the given orbit angle.

        Parameters
        ----------
        targetOrbitAngle : `float`
            The desired orbit angle in radians.

        Raises
        ------
        ConvergenceError
            Raised when the function does not converge to a solution.

        Returns
        -------
        list
            Times corresponding to the desired orbit angle.
        list
            Indeces of the state vector array corresponding to the orbit angle.

        """
        """ Load state vectors for envision """ 
        sv = loadSV(toPCR = False)

        """ Get the period """
        venSAR = orbit(planet=sv.planet, angleUnits="radians")

        """ Initiate some variables """
        idx = 0
        N = len(sv.measurementTime)
        times = []
        idxarr = []

        """ Define a function to compute the time and closest index to
            the given seed idx where the orbitAngle equals the target
            orbit angle. """
        def findClosestIndex(idx, maxIter = 10):
            nIter = 0
            while nIter < maxIter:
                orbitAngle, ascendingNode = venSAR.setFromStateVector(sv.measurementData[idx])
                t1 = venSAR.computeT(targetOrbitAngle)
                t2 = venSAR.computeT(orbitAngle)
                dtasc = np.angle(np.exp(1j*(t1-t2)/venSAR.period))*venSAR.period
                tasc = sv.measurementTime[idx] + np.timedelta64(int(dtasc), 's')
                if tasc < sv.measurementTime[0]:
                    tasc += np.timedelta64(int(venSAR.period), 's')
                newidx = findNearest(sv.measurementTime, tasc)
                if newidx == idx:
                    return idx, dtasc
                else:
                    idx = newidx
                nIter += 1
                
            raise ConvergenceError("findClosestIndex", maxIter)
            
        """ Iterate and find all times corresonding to the target orbit angle """
        nPoints = 0
        while idx < N:
            try:
                new_idx, dtasc = findClosestIndex(idx)
                tasc = sv.measurementTime[new_idx] + np.timedelta64(int(dtasc*1e6), 'us')
                times.append(tasc)
                idxarr.append(new_idx)
                new_idx = findNearest(sv.measurementTime,
                                      sv.measurementTime[new_idx] 
                                      + np.timedelta64(int(venSAR.period), 's'))
                if new_idx == idx:
                    raise ConvergenceError("estimateOrbitAngleTimes", 0)
                else:
                    idx = new_idx
                    
                nPoints += 1
                if nPoints%1000 == 1:
                    print(new_idx)
            except IndexError:
                idx = N + 1
                
        return times, idxarr
    
    def estimateOrbitStartTimes(self, 
                                orbitStartAngle = np.pi/2,
                                minute_err = 0.002,
                                idx_err = 10):
        """
        Compute the start times and indeces of each orbit. Also compute the
        start time of each cycle.

        Parameters
        ----------
        orbitStartAngle: `float`, optional
            The angle from the Ascending node that defines the start of an
            orbit.
        minute_err : 'float', optional
            Timing error threhold to find cycle jumps. The default is 0.002.
        idx_err : `int`, optional
            Indexing threhold to find cycle jumps. The default is 10.

        Returns
        -------
        None.

        """
        self.__tascarr, self.__idxarr = self.estimateOrbitAngleTimes(orbitStartAngle)
        cj = np.argwhere(np.diff(np.diff(self.__tascarr)/np.timedelta64(1,'m')) 
                         > minute_err)[:,0] + 1
        idx = np.argwhere(np.diff(cj)<idx_err)
        self.__cycle_jump = [cj[k] for k in range(len(cj)) if k not in idx]
        return

    def getCycleIdx(self):
        return self.__cycle_jump
    
    def getCycleFromOrbit(self, orbitNumber):
        """
        Get the cycle for given orbit number

        There are 5000-6000 orbits per cycle. This function returns the 
        cycle from the provided orbit number
        
        Parameters
        ----------
        orbitNumber : `int`
            The orbit number.

        Returns
        -------
        `int`
            The computed cycle number.

        """
        
        if orbitNumber < 0 or orbitNumber > self.__idxarr[-1]:
            return None
        for k,cyc in enumerate(self.__cycle_jump):
            if orbitNumber < cyc:
                return k + 1
        return len(self.__cycle_jump) + 1
        
    
    def getCycleFromTime(self, utc):
        if utc < self.__tascarr[0] or utc > self.__tascarr[-1]:
            return None
        for k,cyc in enumerate(self.__cycle_jump):
            if utc < self.__tascarr[cyc]:
                return k + 1
        return len(self.__cycle_jump) + 1
    
    # Function to find the orbit number
    def getOrbitNumber(self, times):
        nearestOrbit = [findNearest(self.__tascarr, t) for t in times]
        branch = [int(t > self.__tascarr[k]) for k,t in zip(nearestOrbit, times)]
        return [nO + br for nO, br in zip(nearestOrbit, branch)]

    def orbitPointIndex(self, 
                        orbitNumber,
                        point):
        """
        Get the index of the state vector on a given orbit that is closest to 
        the given point

        Parameters
        ----------
        orbitNumber : `int`
            The orbit number.
        point : `dict`
            Dictionary representation of the point.

        Returns
        -------
        `int`
            The index of the state vector file on the given orbit that is
            closest to the given point.

        """
        start_idx = self.__idxarr[orbitNumber-1]
        end_idx = self.__idxarr[orbitNumber]
        satX = np.array(self.__sv.measurementData[start_idx:end_idx])[:,:3]
        X = np.array(point["properties"]["target"]["xyz"])
        cidx = np.argmin(np.linalg.norm(satX - np.tile(X, (len(satX),1)), axis=1))
        return self.__idxarr[orbitNumber-1] + cidx 

    # Compute the satellite position k orbits later
    def incidenceZeroDoppler(self, 
                             orbitNumber,
                             point):
        """
        Compute imaging parameters for the given point in the Zero Doppler
        geometry on the given orbit.
        
        This function relies on computeImagingGeometry, a function that 
        propagates the orbit to the right time for Zero-Doppler imaging.

        Parameters
        ----------
        orbitNumber : `int`
            The orbit number.
        point : `dict`
            Dictionary representation of point of interest.

        Returns
        -------
        spoint : `dict`
            Dictionary description of the imaging geometry. This description 
            includes the incidence angle and the satellite position when 
            imaging in a number of reference frames. It also includes the
            range and rdotv (which relates to the Zero Doppler value).

        """
        try:
            X = np.array(point["properties"]["target"]["xyz"])
        except KeyError:
            X = self.llh2xyz(point)
            point["properties"]["target"]["xyz"] = X.tolist()
        
        try:
            N = np.array(point["properties"]["target"]["normal"])
        except KeyError:
            N = self.surfaceNormal(point)
            point["properties"]["target"]["normal"] = N.tolist()
            
        ridx = self.orbitPointIndex(orbitNumber, point)
            
        targetSVtime = self.__sv.measurementTime[ridx]
        R, inc, satSV, err = computeImagingGeometry(self.__sv, 
                                                    targetSVtime, 
                                                    X, 
                                                    N)
        dopError = R.dot(satSV[1][3:])
        satSVME = self.__sv.planet.PCRtoME2000(*satSV)
        satSVICRF = self.__sv.planet.PCRtoICRF(*satSV)
    
        spoint = {"type": "Feature",
                  "geometry": {
                      "type": "Point",
                      "coordinates": list(map(self.__sv.xyz2polar(satSV[1][:3]).__getitem__, 
                                              [1,0,2]))
                      },
                  "properties": {
                      "object": "VenSAR",
                      "orbitNumber": int(orbitNumber),
                      "cycle": self.getCycleFromOrbit(orbitNumber),
                      "orbitDirection": "Ascending" if satSV[1][-1] > 0 else "Descending",
                      "incidence": inc,
                      "range": np.linalg.norm(R),
                      "rdotv": dopError,
                      "stateVector": {
                          "IAU_VENUS": {
                              "time": satSV[0].astype(str),
                              "xyzVxVyVz": satSV[1].tolist()
                              },
                          "VME2000": {
                              "time": satSVME[0].astype(str),
                              "xyzVxVyVz": satSVME[1].tolist()
                              },
                          "J2000": {
                              "time": satSVICRF[0].astype(str),
                              "xyzVxVyVz": satSVICRF[1].tolist()
                              }
                          },
                      "processingTimestamp": np.datetime64("now").astype(str),
                      "targetID": point["properties"]["targetID"]
                      }
                  }
        
        return spoint
    
    def computeSeedPoints(self, 
                          point, 
                          distance = 20000,
                          boundaryThreshold = 1e6,
                          makeplot = False):
        """
        Function to compute points in the state vector file where there is a
        local minimum in range to the given point.

        Parameters
        ----------
        point : `dict`
            Dictionary representing the point.
        distance : `float`, optional
            Distance parameter to pass to the find_peaks function. The default 
            is 20000. Local peaks must be at least this distance apart in 
            samples
        boundaryThreshold : `float`, optional
            Threshold used to test for peaks near the boundaries of the state
            vector file. The default is 1e6.
        makeplot : `bool`, optional
            Flag to generate a plot of the solution. The default is False.

        Returns
        -------
        times : `list`
            List of times of local minima in state vector file.

        """
        """ Get a pointer to the state vector """
        sv = self.__sv
               
        """ Get a 3-D representation of the point """
        try:
            X = np.array(point["properties"]["target"]["xyz"])
        except KeyError:
            X = self.llh2xyz(point)
            point["properties"]["target"]["xyz"] = X.tolist()
            
        """ Compute the range vector to X for all state vectors """
        rvec = np.array([s[:3] - X for s in sv.measurementData])
        rngs = np.linalg.norm(rvec, axis=1)
        vels = np.array([s[3:] for s in sv.measurementData])
        rdotv = np.sum(rvec*vels, axis=1)
        rdotv /= np.linalg.norm(vels, axis=1)
        T = rngs
        
        lm = find_peaks(-T, distance = distance)
        lmm = find_peaks(-T[lm[0]])
        idx = list(lm[0][lmm[0]])
        if T[lm[0][0]] < boundaryThreshold and lm[0][0] not in idx:
            idx = [lm[0][0]] + idx
        if T[lm[0][-1]] < boundaryThreshold and lm[0][-1] not in idx:
            idx = idx + [lm[0][-1]]
        y = [T[k] for k in idx]
        
        if makeplot:
            plt.figure()
            plt.plot(T)
            plt.plot(lm[0], T[lm[0]], 'r.')
            plt.plot(idx, y, 'gs')
        
        times = [sv.measurementTime[i] for i in idx]
        
        return times
        
    def surfaceNormal(self, point):
        X = self.llh2xyz(point)
        return sNorm(X, self.__sv)
    
    def llh2xyz(self, point):
        return self.__sv.llh2xyz([point["geometry"]["coordinates"][k] 
                                  for k in [1,0,2]])
    
    def getSV(self):
        return self.__sv
    
    def computeIncidences(self,
                          initOrbit,
                          targetIncidenceStart = 22,
                          targetIncidenceEnd = 36,
                          dIncidence = 2.0,
                          maxIter=100):
        pointAnalysis = {
            "type": "FeatureCollection",
            "features": []
            }
        
        guess = 1
        infLoop = 0
        nOrbits = len(self.__tascarr)
        while guess != 0:
            spoint = self.incidenceZeroDoppler(initOrbit, point)
            guess = int((spoint["properties"]["incidence"]
                         - targetIncidenceStart)/dIncidence)
            sgn = 1 if spoint["properties"]["orbitDirection"] == "Ascending" else -1
            initOrbit += sgn*guess
            if initOrbit < 0 or initOrbit > nOrbits:
                raise BoundsError(initOrbit, 0, nOrbits)
            infLoop += 1
            if infLoop > maxIter:
                raise ConvergenceError("Find targetIncidenceStart", maxIter)
        
        pointAnalysis["features"].append(spoint)
        increment = -sgn*np.sign(targetIncidenceStart)
        infLoop = 0
        while np.abs(spoint["properties"]["incidence"]) < np.abs(targetIncidenceEnd): 
            initOrbit += increment
            spoint = self.incidenceZeroDoppler(initOrbit, point)
            pointAnalysis["features"].append(spoint)
            infLoop += 1
            if infLoop > maxIter:
                raise ConvergenceError("Iterate to targetIncidenceEnd", maxIter)
        return pointAnalysis


#%% Get an instance of the envisionIncidence class
eI = envisionIncidence()

#%% Read ROI file
if 'linux' in sys.platform:
    filepath = '/users/isikanet/local/data/cycles'
else:
    filepath = os.path.join(r"C:\Users",
                            r"ishuwa.sikaneta",
                            r"OneDrive - ESA",
                            r"Documents",
                            r"ESTEC",
                            r"Envision",
                            r"ROIs",
                            r"Plan")

with open(os.path.join(filepath, "roi.geojson"), "r") as f:
    featureCollection = json.loads(f.read())
    
#%% Loop through points
for polygon in featureCollection["features"]:
    logger.info("ROI: %s" % polygon["properties"]["ROI_No"])
    point = { 
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": list(np.mean(polygon["geometry"]["coordinates"][0][0:-1], axis=0))
        },
        "properties": {
            "object": "target",
            "targetID": polygon["properties"]["ROI_No"],
            "target": {}
        }
    }
    
    """ Compute coordinates of center of roI """
    X = eI.llh2xyz(point)
    point["properties"]["target"]["xyz"] = X.tolist()
    
    """ Write point to file """
    filename = "%s_%s.geojson" % (polygon["properties"]["ROI_No"], "center")
    with open(os.path.join(filepath, filename), "w") as f:
        f.write(json.dumps(point, indent=2))    
        
    """ Loop through cycles """
    pointAnalysis = {
        "type": "FeatureCollection",
        "features": []
        }
    
    tSeed = eI.computeSeedPoints(point)
    my_orbits = eI.getOrbitNumber(tSeed)
    iAngles = [(22,36), (-22,-36)]
    for my_orbit in tqdm(my_orbits):
        for iAngle in iAngles:
            try:
                pointAnalysis["features"] += eI.computeIncidences(my_orbit, 
                                                                  iAngle[0], 
                                                                  iAngle[1])["features"]
            except ConvergenceError:
                logger.warning("Convergence Error: ROI: %s, orbitNumber: %d, incidences: [%d, %d]" % 
                               (polygon["properties"]["ROI_No"],
                                my_orbit,
                                iAngle[0],
                                iAngle[1]))
            except BoundsError as bE:
                logger.warning("ROI: %s Bounds Error: %s" % (polygon["properties"]["ROI_No"],
                                                             str(bE)))
            except IndexError:
                logger.warning("Index Error: ROI: %s, orbitNumber: %d, incidences: [%d, %d]" % 
                               (polygon["properties"]["ROI_No"],
                                my_orbit,
                                iAngle[0],
                                iAngle[1]))
                
    
    """ Write point to file """
    filenameInc = "%s_%s.geojson" % (polygon["properties"]["ROI_No"], 
                                     "incidence")
    with open(os.path.join(filepath, filenameInc), "w") as f:
        f.write(json.dumps(pointAnalysis, indent=2))\
