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
        return '(%s) did not converge after %d iterations' % (self.algorithm, self.maxiter)

#%% Define class to capture convergence errors
class BoundsError(Exception):
    def __init__(self, predicted, lowerBound, upperBound):
        self.index = predicted
        self.lowerBound = lowerBound
        self.upperBound = upperBound
    def __str__(self):
        return 'Value: %d outside of set [%d, %d]' % (self.index, self.lowerBound, self.upperBound)
        
#%%
class envisionIncidence:
    
    def __init__(self, do_plots = False):
        self.plots = do_plots
        self.__tascarr = []
        self.__idxarr = []
        self.__sv = None
        self.__cycle_jump = []
        self.estimateAscendingTimes()#%% Reload state vectors but in VCR frame
        self.__sv = loadSV(toPCR = True)
    
    def getApproximateAscNodeTimes(self):
        return self.__tascarr, self.__idxarr
    
    def estimateOrbitAngleTimes(self, targetOrbitAngle):
        """ Load state vectors for envision """ 
        sv = loadSV(toPCR = False)

        """ Get the period """
        venSAR = orbit(planet=sv.planet, angleUnits="radians")

        """ Initiate some variables """
        idx = 0
        N = len(sv.measurementTime)
        times = []
        idxarr = []

        """ Define a function to compute the time and closest index
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
    
    def estimateAscendingTimes(self):
        self.__tascarr, self.__idxarr = self.estimateOrbitAngleTimes(np.pi/2)
        # # # Load state vectors for envision
        # # sv = loadSV(toPCR = False)

        # # # Get the period
        # # venSAR = orbit(planet=sv.planet, angleUnits="radians")

        # # # Estimate ascending Node times
        # # idx = 0
        # # N = len(sv.measurementTime)
        # # self.__tascarr = []
        # # self.__idxarr = []

        # # """ Iterate and find all asceding node crossing times """
        # # while idx < N:
        # #     orbitAngle, ascendingNode = venSAR.setFromStateVector(sv.measurementData[idx])
        # #     dtasc = venSAR.period + venSAR.computeT(0)-venSAR.computeT(orbitAngle)
        # #     tasc = sv.measurementTime[idx] + np.timedelta64(int(dtasc), 's')
        # #     self.__tascarr.append(tasc)
        # #     try:
        # #         idx = findNearest(sv.measurementTime, tasc)
        # #         self.__idxarr.append(idx)
        # #         if idx%1000 == 1:
        # #             print(idx)
        # #     except IndexError:
        # #         idx = N + 1

        # # Check the time differences
        # if 'linux' not in sys.platform and self.plots:
        #     kep = [venSAR.state2kepler(sv.measurementData[k]) for k in self.__idxarr]
        #     z = [sv.measurementData[k][2] for k in self.__idxarr]
        #     plt.figure()
        #     plt.plot(self.__tascarr[0:-1], z)
        #     plt.grid()
        #     plt.show()
            
        #     period = [2*np.pi*np.sqrt(venSAR.state2kepler(sv.measurementData[k])['a']**3
        #                               /venSAR.planet.GM) for k in self.__idxarr]
        #     plt.figure()
        #     plt.plot(self.__tascarr[0:-1], period)
        #     plt.xlabel('Time')
        #     plt.ylabel('Orbit Period (s)')
        #     plt.title('Orbit period for ET1 2031 N')
        #     plt.grid()
        #     plt.show()
            
            
        #     a = [venSAR.state2kepler(sv.measurementData[k])['a']/1e3 for k in self.__idxarr]
        #     plt.figure()
        #     plt.plot(self.__tascarr[0:-1], a)
        #     plt.xlabel('Time')
        #     plt.ylabel('Orbit semi-major axis (km)')
        #     plt.title('Orbit semi-major axis for ET1 2031 N')
        #     plt.grid()
        #     plt.show()
            
        #     plt.figure()
        #     plt.plot(self.__tascarr[0:-1], np.diff(self.__tascarr)/np.timedelta64(1,'m'))
        #     plt.grid()
        #     plt.title('Orbit period (minutes)')
        #     plt.xlabel('Time')
        #     plt.ylabel('Orbit period (minutes)')
        #     plt.show()
            
        #     perigee = [k["perigee"] for k in kep]
        #     plt.figure()
        #     plt.plot(self.__tascarr[0:-1], np.degrees(np.unwrap(perigee)))
        #     plt.xlabel('Time')
        #     plt.ylabel('Orbit perigee angle (deg)')
        #     plt.title('Orbit perigee angle for ET1 2031 N')
        #     plt.grid()
        #     plt.show()
        
        #
        cj = np.argwhere(np.diff(np.diff(self.__tascarr)/np.timedelta64(1,'m')) > 0.002)[:,0] + 1
        idx = np.argwhere(np.diff(cj)<10)
        self.__cycle_jump = [cj[k] for k in range(len(cj)) if k not in idx]
        return

    def getCycleIdx(self):
        return self.__cycle_jump
    
    def getCycleFromOrbit(self, orbitNumber):
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

    # def plotIncidenceFigure(self, point):
    #     symbols = ['.', 'o', 'x', 'd', 's']
    #     groundPoint = tuple(point["target"]["llh"])
    #     targetInc = point["target"]["satReference"]["incidence"]
    #     plt.figure()
    #     myincidences = [[p["incidence"]
    #                      for p in cycle] 
    #                     for cycle in point["cycle"]]
        
    #     for data,symbol in zip(myincidences, symbols):
    #         plt.plot(range(len(data)), data, symbol)
            
    #     plt.grid()
    #     x_axis = np.arange(len(myincidences[0]))
    #     y_axis = np.ones_like(x_axis)*targetInc
    #     plt.plot(range(len(data)), y_axis)
    #     plt.xlabel('orbit')
    #     plt.ylabel('Incidence Angle (deg)')
    #     plt.legend(['cycle 2', 'cycle 3', 'cycle 4', 'cycle 5', 'cycle 6', 'cycle 1'])
    #     plt.title('Incidence to point lat: %0.2f (deg), lon %0.2f (deg), hae: %0.2f (m)' % groundPoint)
    #     plt.show()
        
    #     return plt

    # def plotSpiderFigure(self, point):
    #     oNS = np.arange(5)
    #     groundPoint = tuple(point["target"]["llh"])
    #     myincidences = [[p["incidence"]
    #                      for p in cycle] 
    #                     for cycle in point["cycle"]]
        
    #     myorbitNum = [[p["orbitNumber"]%5 for p in cycle] 
    #                   for cycle in point["cycle"]]
        
    #     pl = [[[x for x,k in zip(myinc, myorb) if k == n] 
    #             for n in oNS] for myinc, myorb in zip(myincidences, myorbitNum)]
        
    #     # Compute the orbit number on cycle 1
    #     refOrbitNumber = point["target"]["satReference"]["orbitNumber"]
    #     targetInc = point["target"]["satReference"]["incidence"]
        
    #     # Plot the incidence differences
    #     lblstr = 'lat: %0.2f (deg), lon %0.2f (deg), hae: %0.2f (m)' % groundPoint
    #     cplot = [(0, '1b'),
    #              (1, 'xr'),
    #              (2, '+g'),
    #              (3, '*k'),
    #              (4, '2m')]
        
    #     fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        
    #     for cycleNum, symbol in cplot:
    #         lbl = "Cycle %d" % (cycleNum + 2)
    #         for k in range(5):
    #             r = np.array(pl[cycleNum][k]) - targetInc
    #             th = np.angle(np.exp(k*1j*np.pi/5)*r)
    #             if k>0:
    #                 ax.plot(th, 
    #                         np.abs(r), 
    #                         symbol)
    #             else:
    #                 ax.plot(th, 
    #                         np.abs(r), 
    #                         symbol,
    #                         label = lbl)
            
    #     ax.set_rticks([1.5, 4, 7]) 
    #     ax.set_rlim([0,7])
    #     xT = np.arange(0,2*np.pi,np.pi/5)
    #     xL = ['orbit 1', 
    #           'orbit 2', 
    #           'orbit 3', 
    #           'orbit 4', 
    #           'orbit 5', 
    #           'orbit 1', 
    #           'orbit 2', 
    #           'orbit 3', 
    #           'orbit 4', 
    #           'orbit 5']
    #     plt.xticks(xT, xL)
    #     plt.title("Ref orbit: %d, incidence %0.4f (deg)\n%s" % (refOrbitNumber, 
    #                                                             targetInc, 
    #                                                             lblstr))
    #     plt.legend()
    #     plt.show()
        
    #     return plt
    
    # def matchIncidencePermute(self,
    #                           point, 
    #                           incidence_range = [23,37],
    #                           criteria = [(0, 1.5), (5.5, 1.5)]):
    #     orbitNumber = point["target"]["satReference"]["orbitNumber"]
    #     oem = orbitNumber%5
    #     flt_cyc = [[p for p in cyc if p["incidence"] < incidence_range[1] 
    #                 and p["incidence"] > incidence_range[0]
    #                 and p["orbitNumber"]%5 == oem] 
    #                 for cyc in point["cycle"]]
    #     #if any([1 for f in flt_cyc if len(f) == 0]):
    #     orbit_fix = {"append": np.arange(1,21),
    #                  "prepend": -np.arange(1,21)}
    #     for k, fl in enumerate(flt_cyc):
    #         if len(fl)== 0:
    #             i_vals = [x["incidence"] for x in point["cycle"][k]]
    #             i_max = max(i_vals)
    #             i_min = min(i_vals)
    #             slope = np.sign(i_vals[2]-i_vals[0])
    #             operation = "append"
    #             if i_min > incidence_range[0]:
    #                 if slope < 0:
    #                     """ append values """
    #                     pass
    #                 else:
    #                     """ prepend values """
    #                     operation= "prepend"
    #             elif i_max < incidence_range[1]:
    #                 if slope < 0:
    #                     """ prepend values """
    #                     operation = "prepend"
    #                 else:
    #                     """ append values """
    #                     pass
    #             else:
    #                 print("I don't know what to do! Cycle %d" % (k+2))
    #                 print(40*"=")
    #                 print("Orbit: %d Slope %f" % (k, slope))
    #                 print("Incidence %f,%f" % (i_min, i_max))
    #                 print(40*"=")
    #                 raise ValueError
                        
    #             """ Assume that we found the right operation """
    #             print("Orbit: %d, Cycle: %d needs a(n) %s" % (orbitNumber,
    #                                                           k+2,
    #                                                           operation))
    #             additional_orbits = self.extend(point, 
    #                                             self.__sv, 
    #                                             k, 
    #                                             orbitRange = orbit_fix[operation])
    #             if operation == "append":
    #                 point["cycle"][k] = point["cycle"][k] + additional_orbits
    #             else:
    #                 point["cycle"][k] = additional_orbits + point["cycle"][k]
                    
    #             flt_cyc = [[p for p in cyc if p["incidence"] < incidence_range[1] 
    #                         and p["incidence"] > incidence_range[0]
    #                         and p["orbitNumber"]%5 == oem] 
    #                         for cyc in point["cycle"]]
            
    #     reference = {"incidence": point["target"]["satReference"]["incidence"],
    #                  "orbitNumber": point["target"]["satReference"]["orbitNumber"]}
    #     perms = [[reference,v,w,x,y,z] 
    #               for v in flt_cyc[0] 
    #               for w in flt_cyc[1] 
    #               for x in flt_cyc[2]
    #               for y in flt_cyc[3]
    #               for z in flt_cyc[4]]
        
    #     upper = np.array([[0,1,1,1,1,1],
    #                       [0,0,1,1,1,1],
    #                       [0,0,0,1,1,1],
    #                       [0,0,0,0,1,1],
    #                       [0,0,0,0,0,1],
    #                       [0,0,0,0,0,0]])
    #     lower = upper.T
    #     orbNumbers = []
    #     incidences = []
    #     scores = []
    #     for perm in perms:
    #         x = np.array([p["incidence"] for p in perm])
    #         incidences.append(x)
    #         orbNumbers.append(np.array([p["orbitNumber"] for p in perm]))
    #         score = []
    #         for offset, threshold in criteria:
    #             y = np.abs((np.tile(x, 6).reshape((6,6)) - 
    #                         np.tile(x, 6).reshape((6,6)).T - offset)/threshold)
                
    #             uVals = y*upper
    #             lVals = y*lower
    #             uScore = (uVals>0).astype(int) * (uVals<1).astype(int)
    #             lScore = (lVals>0).astype(int) * (lVals<1).astype(int)
    #             score.append(np.sum((uScore + lScore.T > 0).astype(int)))
    #         scores.append(score)
            
    #     return np.array(scores), np.array(incidences), np.array(orbNumbers), point
       
    # def extend(self,
    #            point, 
    #            cycle_num,
    #            orbitRange = np.arange(1,21)):
        
    #     cycle = point["cycle"][cycle_num]
        
    #     r_cycle = cycle[-1] if orbitRange[0]>0 else cycle[0]
        
    #     # Compute the orbit number on reference cycle
    #     refOrbitNumber = r_cycle["orbitNumber"]
    #     refTime = np.datetime64(r_cycle["state_vector"]["time"])
    #     refSV = np.array(r_cycle["state_vector"]["satpos"] + 
    #                      r_cycle["state_vector"]["satvel"])
        
    #     xG = np.array(point["target"]["xyz"])
    #     xG_snormal = np.array(point["target"]["normal"])
        
    #     """ The next two lines are to reset the period """
    #     myOrbit = orbit(planet=self.__sv.planet, angleUnits="radians")
    #     orbitAngle, ascendingNode = myOrbit.setFromStateVector(refSV)
    #     period = np.timedelta64(int(myOrbit.period*1e6),'us')
        
    #     """ Generate an array of times to examine """
    #     etarange = refTime + orbitRange*period
        
    #     """ Compute the imaging geometry to broadside at these times """
    #     try:
    #         options = [computeImagingGeometry(self.__sv, eta, xG, xG_snormal) 
    #                    for eta in etarange]
    #     except IndexError:
    #         logger.warning("Failed extend for cycle : %d" % cycle_num)
        
    #     """ Compute the orbit number """
    #     orbitNumber = refOrbitNumber + orbitRange
        
    #     """ Write to json/dict """
    #     add_orbits = [
    #              {
    #               "incidence": o[1],
    #               "range": np.linalg.norm(o[0]),
    #               "orbitNumber": int(oN),
    #               "state_vector": 
    #                   {
    #                    "time": str(np.datetime_as_string(o[2][0])),
    #                    "satpos": o[2][1][:3].tolist(),
    #                    "satvel": o[2][1][3:].tolist()
    #                   },
    #               "llh": list(self.__sv.xyz2polar(o[2][1][:3]))
    #              } for o, oN in zip(options, orbitNumber)
    #             ]
        
    #     return add_orbits

    #
    def orbitPointIndex(self, 
                        orbitNumber,
                        point):
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
        R, inc, satSV, err = computeImagingGeometry(self.__sv, targetSVtime, X, N)
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
        # Find the nearest state vectors to X
        sv = self.__sv
               
        # Find the minimum distance state vector over the cycle
        try:
            X = np.array(point["properties"]["target"]["xyz"])
        except KeyError:
            X = self.llh2xyz(point)
            point["properties"]["target"]["xyz"] = X.tolist()
            
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
        
        
    def cyclePointOrbitNumber(self, cycle, point):              
        # Find the nearest state vector to X
        cycle_jump = self.__cycle_jump
        idxarr = self.__idxarr
        sv = self.__sv
        
        start_idxs = [0] + [x+1 for x in cycle_jump]
        end_idxs = list(cycle_jump) + [len(idxarr)-1]
        
        start_idx = idxarr[start_idxs[cycle-1]]
        end_idx = idxarr[end_idxs[cycle-1]]
        
        # cycX = np.array(self.__sv.measurementData[start_idx:end_idx])[:,:3]
        cycX = sv.measurementData[start_idx:end_idx]
        
        # Find the minimum distance state vector over the cycle
        try:
            X = np.array(point["properties"]["target"]["xyz"])
        except KeyError:
            X = self.llh2xyz(point)
            point["properties"]["target"]["xyz"] = X.tolist()
        
        rvec = np.array([s[:3] - X for s in cycX])
        rngs = np.linalg.norm(rvec, axis=1)
        vels = np.array([s[3:] - X for s in cycX])
        rdotv = np.sum(rvec*vels, axis=1)
        rdotv /= np.linalg.norm(vels, axis=1)
        
        rngsAsc = [r if s[-1]>0 else np.nan for r,s in zip(rngs, cycX)]
        rngsDsc = [r if s[-1]<0 else np.nan for r,s in zip(rngs, cycX)]
        
        idxAsc = start_idx + np.nanargmin(rngsAsc)
        idxDsc = start_idx + np.nanargmin(rngsDsc)
        
        #cidx = np.argmin(np.linalg.norm(cycX 
        #                                - np.tile(X, (len(cycX),1)), axis=1))
        
        #cidx += start_idx
        
        # Get the orbit number
        return (self.getOrbitNumber([sv.measurementTime[idxAsc]])[0],
                self.getOrbitNumber([sv.measurementTime[idxDsc]])[0])
        
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
rootfolder = r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\Envision\ROIs\PlanB"
with open(os.path.join(rootfolder, "roi.geojson"), "r") as f:
    featureCollection = json.loads(f.read())

#%% Define the path for writing data
if 'linux' in sys.platform:
    filepath = '/users/isikanet/local/data/cycles'
else:
    filepath = os.path.join(rootfolder, "incidence")
    
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
    filenameInc = "%s_%s.geojson" % (polygon["properties"]["ROI_No"], "incidence")
    with open(os.path.join(filepath, filenameInc), "w") as f:
        f.write(json.dumps(pointAnalysis, indent=2))\
