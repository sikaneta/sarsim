# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 14:22:02 2022

@author: ishuwa.sikaneta
"""

from orbit.geometry import getTiming
from orbit.geometry import surfaceNormal
from orbit.geometry import findNearest
from orbit.geometry import computeImagingGeometry
from orbit.orientation import orbit
from orbit.envision import loadSV
from measurement.measurement import state_vector

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from glob import glob
import sys

#%%
class envisionIncidence:
    
    def __init__(self):
        #pass
        
        self.__tascarr = []
        self.__idxarr = []
        self.__sv = None
        self.__cycle_jump = []
        self.estimateAscendingTimes()#%% Reload state vectors but in VCR frame
        self.__sv = loadSV(toPCR = True)
    
    def estimateAscendingTimes(self):
        # Load state vectors for envision
        sv = loadSV(toPCR = False)

        # Get the period
        venSAR = orbit(planet=sv.planet, angleUnits="radians")

        # Estimate ascending Node times
        idx = 0
        N = len(sv.measurementTime)
        self.__tascarr = []
        self.__idxarr = []

        """ Iterate and find all asceding node crossing times """
        while idx < N:
            orbitAngle, ascendingNode = venSAR.setFromStateVector(sv.measurementData[idx])
            dtasc = venSAR.period + venSAR.computeT(0)-venSAR.computeT(orbitAngle)
            tasc = sv.measurementTime[idx] + np.timedelta64(int(dtasc), 's')
            self.__tascarr.append(tasc)
            try:
                idx = findNearest(sv.measurementTime, tasc)
                self.__idxarr.append(idx)
                if idx%1000 == 1:
                    print(idx)
            except IndexError:
                idx = N + 1

        # Check the time differences
        if 'linux' not in sys.platform and 0:
            kep = [venSAR.state2kepler(sv.measurementData[k]) for k in self.__idxarr]
            z = [sv.measurementData[k][2] for k in self.__idxarr]
            plt.figure()
            plt.plot(self.__tascarr[0:-1], z)
            plt.grid()
            plt.show()
            
            period = [2*np.pi*np.sqrt(venSAR.state2kepler(sv.measurementData[k])['a']**3
                                      /venSAR.planet.GM) for k in self.__idxarr]
            plt.figure()
            plt.plot(self.__tascarr[0:-1], period)
            plt.xlabel('Time')
            plt.ylabel('Orbit Period (s)')
            plt.title('Orbit period for ET1 2031 N')
            plt.grid()
            plt.show()
            
            
            a = [venSAR.state2kepler(sv.measurementData[k])['a']/1e3 for k in self.__idxarr]
            plt.figure()
            plt.plot(self.__tascarr[0:-1], a)
            plt.xlabel('Time')
            plt.ylabel('Orbit semi-major axis (km)')
            plt.title('Orbit semi-major axis for ET1 2031 N')
            plt.grid()
            plt.show()
            
            plt.figure()
            plt.plot(self.__tascarr[0:-1], np.diff(self.__tascarr)/np.timedelta64(1,'m'))
            plt.grid()
            plt.title('Orbit period (minutes)')
            plt.xlabel('Time')
            plt.ylabel('Orbit period (minutes)')
            plt.show()
            
            perigee = [k["perigee"] for k in kep]
            plt.figure()
            plt.plot(self.__tascarr[0:-1], np.degrees(np.unwrap(perigee)))
            plt.xlabel('Time')
            plt.ylabel('Orbit perigee angle (deg)')
            plt.title('Orbit perigee angle for ET1 2031 N')
            plt.grid()
            plt.show()
        
        #
        self.__cycle_jump = np.argwhere(np.diff(np.diff(self.__tascarr)/np.timedelta64(1,'m')) > 0.02)[:,0] + 1
        
        return



    # Function to find the orbit number
    def getOrbitNumber(self, times):
        nearestOrbit = [findNearest(self.__tascarr, t) for t in times]
        # orbitNumber = [oN if t > timeArray[oN] else oN-1 
        #                for oN, t in zip(nearestOrbit,times)]
        # return orbitNumber
        return nearestOrbit

    def plotIncidenceFigure(self, point):
        symbols = ['.', 'o', 'x', 'd', 's']
        groundPoint = tuple(point["target"]["llh"])
        targetInc = point["target"]["satReference"]["incidence"]
        plt.figure()
        myincidences = [[p["incidence"]
                         for p in cycle] 
                        for cycle in point["cycle"]]
        
        for data,symbol in zip(myincidences, symbols):
            plt.plot(range(len(data)), data, symbol)
            
        plt.grid()
        x_axis = np.arange(len(myincidences[0]))
        y_axis = np.ones_like(x_axis)*targetInc
        plt.plot(range(len(data)), y_axis)
        plt.xlabel('orbit')
        plt.ylabel('Incidence Angle (deg)')
        plt.legend(['cycle 2', 'cycle 3', 'cycle 4', 'cycle 5', 'cycle 6', 'cycle 1'])
        plt.title('Incidence to point lat: %0.2f (deg), lon %0.2f (deg), hae: %0.2f (m)' % groundPoint)
        plt.show()
        
        return plt

    def plotSpiderFigure(self, point):
        oNS = np.arange(5)
        groundPoint = tuple(point["target"]["llh"])
        myincidences = [[p["incidence"]
                         for p in cycle] 
                        for cycle in point["cycle"]]
        
        myorbitNum = [[p["orbitNumber"]%5 for p in cycle] 
                      for cycle in point["cycle"]]
        
        pl = [[[x for x,k in zip(myinc, myorb) if k == n] 
                for n in oNS] for myinc, myorb in zip(myincidences, myorbitNum)]
        
        # Compute the orbit number on cycle 1
        refOrbitNumber = point["target"]["satReference"]["orbitNumber"]
        targetInc = point["target"]["satReference"]["incidence"]
        
        # Plot the incidence differences
        lblstr = 'lat: %0.2f (deg), lon %0.2f (deg), hae: %0.2f (m)' % groundPoint
        cplot = [(0, '1b'),
                 (1, 'xr'),
                 (2, '+g'),
                 (3, '*k'),
                 (4, '2m')]
        
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        
        for cycleNum, symbol in cplot:
            lbl = "Cycle %d" % (cycleNum + 2)
            for k in range(5):
                r = np.array(pl[cycleNum][k]) - targetInc
                th = np.angle(np.exp(k*1j*np.pi/5)*r)
                if k>0:
                    ax.plot(th, 
                            np.abs(r), 
                            symbol)
                else:
                    ax.plot(th, 
                            np.abs(r), 
                            symbol,
                            label = lbl)
            
        ax.set_rticks([1.5, 4, 7]) 
        ax.set_rlim([0,7])
        xT = np.arange(0,2*np.pi,np.pi/5)
        xL = ['orbit 1', 
              'orbit 2', 
              'orbit 3', 
              'orbit 4', 
              'orbit 5', 
              'orbit 1', 
              'orbit 2', 
              'orbit 3', 
              'orbit 4', 
              'orbit 5']
        plt.xticks(xT, xL)
        plt.title("Ref orbit: %d, incidence %0.4f (deg)\n%s" % (refOrbitNumber, 
                                                                targetInc, 
                                                                lblstr))
        plt.legend()
        plt.show()
        
        return plt
    
    def matchIncidencePermute(self,
                              point, 
                              incidence_range = [23,37],
                              criteria = [(0, 1.5), (5.5, 1.5)]):
        orbitNumber = point["target"]["satReference"]["orbitNumber"]
        oem = orbitNumber%5
        flt_cyc = [[p for p in cyc if p["incidence"] < incidence_range[1] 
                    and p["incidence"] > incidence_range[0]
                    and p["orbitNumber"]%5 == oem] 
                    for cyc in point["cycle"]]
        #if any([1 for f in flt_cyc if len(f) == 0]):
        orbit_fix = {"append": np.arange(1,21),
                     "prepend": -np.arange(1,21)}
        for k, fl in enumerate(flt_cyc):
            if len(fl)== 0:
                i_vals = [x["incidence"] for x in point["cycle"][k]]
                i_max = max(i_vals)
                i_min = min(i_vals)
                slope = np.sign(i_vals[2]-i_vals[0])
                operation = "append"
                if i_min > incidence_range[0]:
                    if slope < 0:
                        """ append values """
                        pass
                    else:
                        """ prepend values """
                        operation= "prepend"
                elif i_max < incidence_range[1]:
                    if slope < 0:
                        """ prepend values """
                        operation = "prepend"
                    else:
                        """ append values """
                        pass
                else:
                    print("I don't know what to do! Cycle %d" % (k+2))
                    print(40*"=")
                    print("Orbit: %d Slope %f" % (k, slope))
                    print("Incidence %f,%f" % (i_min, i_max))
                    print(40*"=")
                    raise ValueError
                        
                """ Assume that we found the right operation """
                print("Orbit: %d, Cycle: %d needs a(n) %s" % (orbitNumber,
                                                              k+2,
                                                              operation))
                additional_orbits = self.extend(point, 
                                                self.__sv, 
                                                k, 
                                                orbitRange = orbit_fix[operation])
                if operation == "append":
                    point["cycle"][k] = point["cycle"][k] + additional_orbits
                else:
                    point["cycle"][k] = additional_orbits + point["cycle"][k]
                    
                flt_cyc = [[p for p in cyc if p["incidence"] < incidence_range[1] 
                            and p["incidence"] > incidence_range[0]
                            and p["orbitNumber"]%5 == oem] 
                            for cyc in point["cycle"]]
            
        reference = {"incidence": point["target"]["satReference"]["incidence"],
                     "orbitNumber": point["target"]["satReference"]["orbitNumber"]}
        perms = [[reference,v,w,x,y,z] 
                  for v in flt_cyc[0] 
                  for w in flt_cyc[1] 
                  for x in flt_cyc[2]
                  for y in flt_cyc[3]
                  for z in flt_cyc[4]]
        
        upper = np.array([[0,1,1,1,1,1],
                          [0,0,1,1,1,1],
                          [0,0,0,1,1,1],
                          [0,0,0,0,1,1],
                          [0,0,0,0,0,1],
                          [0,0,0,0,0,0]])
        lower = upper.T
        orbNumbers = []
        incidences = []
        scores = []
        for perm in perms:
            x = np.array([p["incidence"] for p in perm])
            incidences.append(x)
            orbNumbers.append(np.array([p["orbitNumber"] for p in perm]))
            score = []
            for offset, threshold in criteria:
                y = np.abs((np.tile(x, 6).reshape((6,6)) - 
                            np.tile(x, 6).reshape((6,6)).T - offset)/threshold)
                
                uVals = y*upper
                lVals = y*lower
                uScore = (uVals>0).astype(int) * (uVals<1).astype(int)
                lScore = (lVals>0).astype(int) * (lVals<1).astype(int)
                score.append(np.sum((uScore + lScore.T > 0).astype(int)))
            scores.append(score)
            
        return np.array(scores), np.array(incidences), np.array(orbNumbers), point
       
    def extend(self,
               point, 
               cycle_num,
               orbitRange = np.arange(1,21)):
        
        cycle = point["cycle"][cycle_num]
        
        r_cycle = cycle[-1] if orbitRange[0]>0 else cycle[0]
        
        # Compute the orbit number on reference cycle
        refOrbitNumber = r_cycle["orbitNumber"]
        refTime = np.datetime64(r_cycle["state_vector"]["time"])
        refSV = np.array(r_cycle["state_vector"]["satpos"] + 
                         r_cycle["state_vector"]["satvel"])
        
        xG = np.array(point["target"]["xyz"])
        xG_snormal = np.array(point["target"]["normal"])
        
        """ The next two lines are to reset the period """
        myOrbit = orbit(planet=self.__sv.planet, angleUnits="radians")
        orbitAngle, ascendingNode = myOrbit.setFromStateVector(refSV)
        period = np.timedelta64(int(myOrbit.period*1e6),'us')
        
        """ Generate an array of times to examine """
        etarange = refTime + orbitRange*period
        
        """ Compute the imaging geometry to broadside at these times """
        try:
            options = [computeImagingGeometry(self.__sv, eta, xG, xG_snormal) 
                       for eta in etarange]
        except IndexError:
            print("Failed extend for cycle : %d" % cycle_num)
        
        """ Compute the orbit number """
        orbitNumber = refOrbitNumber + orbitRange
        
        """ Write to json/dict """
        add_orbits = [
                 {
                  "incidence": o[1],
                  "range": np.linalg.norm(o[0]),
                  "orbitNumber": int(oN),
                  "state_vector": 
                      {
                       "time": str(np.datetime_as_string(o[2][0])),
                       "satpos": o[2][1][:3].tolist(),
                       "satvel": o[2][1][3:].tolist()
                      },
                  "llh": list(self.__sv.xyz2polar(o[2][1][:3]))
                 } for o, oN in zip(options, orbitNumber)
                ]
        
        return add_orbits

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
        R, inc, satSV = computeImagingGeometry(self.__sv, targetSVtime, X, N)
        dopError = R.dot(satSV[1][3:])
    
        spoint = {"type": "Feature",
                  "geometry": {
                      "type": "Point",
                      "coordinates": list(map(self.__sv.xyz2polar(satSV[1][:3]).__getitem__, 
                                              [1,0,2]))
                      },
                  "properties": {
                      "object": "VenSAR",
                      "orbitNumber": orbitNumber,
                      "orbitDirection": "Ascending" if satSV[1][-1] > 0 else "Descending",
                      "incidence": inc,
                      "range": np.linalg.norm(R),
                      "rdotv": dopError,
                      "stateVector": { 
                                      "time": satSV[0].astype(str),
                                      "xyzVxVyVz": satSV[1].tolist()
                                 },
                      "processingTimestamp": np.datetime64("now").astype(str),
                      "targetID": point["properties"]["targetID"]
                     }
                  }
        
        return spoint

    def cyclePointOrbitNumber(self, cycle, point):              
        # Find the nearest state vector to X
        start_idxs = [0] + [x+1 for x in self.__cycle_jump]
        end_idxs = list(self.__cycle_jump) + [len(self.__idxarr)-1]
        
        start_idx = self.__idxarr[start_idxs[cycle-1]]
        end_idx = self.__idxarr[end_idxs[cycle-1]]
        
        cycX = np.array(self.__sv.measurementData[start_idx:end_idx])[:,:3]
        
        # Find the minimum distance state vector over the cycle
        try:
            X = np.array(point["properties"]["target"]["xyz"])
        except KeyError:
            X = self.llh2xyz(point)
            point["properties"]["target"]["xyz"] = X.tolist()
        
        cidx = np.argmin(np.linalg.norm(cycX 
                                        - np.tile(X, (len(cycX),1)), axis=1))
        
        cidx += start_idx
        
        # Get the orbit number
        return self.getOrbitNumber([self.__sv.measurementTime[cidx]])[0]
        
    def surfaceNormal(self, point):
        X = self.llh2xyz(point)
        return surfaceNormal(X, self.__sv)
    
    def llh2xyz(self, point):
        return self.__sv.llh2xyz([point["geometry"]["coordinates"][k] for k in [1,0,2]])
    
    def getSV(self):
        return self.__sv

#%% Get an instance of the envisionIncidence class
eI = envisionIncidence()

#%% Read ROI file
with open(r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\Envision\ROIs\roi.geojson", "r") as f:
    featureCollection = json.loads(f.read())


#%% Define the path for writing data
if 'linux' in sys.platform:
    filepath = '/users/isikanet/local/data/cycles'
else:
    filepath = r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\Envision\ROIs\incidence"
    
#%%
polygon = featureCollection["features"][1]

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
# targetIDStr = "https://www.esa.int/%0.2f/%0.2f/%0.1f" % tuple(point["geometry"]["coordinates"])
# # X = eI.llh2xyz(point)
# # N = eI.surfaceNormal(point)
# # point["properties"]["target"] = {"xyz": X.tolist(), 
# #                                  "normal": N.tolist()}
# point["properties"]["targetID"] = str(uuid.uuid3(uuid.NAMESPACE_URL, targetIDStr))




#%% Write point to file
filename = "%s_%s.geojson" % (polygon["properties"]["ROI_No"], "center")
with open(os.path.join(filepath, filename), "w") as f:
    f.write(json.dumps(point, indent=2))
    
#%% Find the closest satellite point
passTypes = ["standard", "stereo", "polarimetry", "highRes"]
for pType in passTypes:
    if pType not in polygon["properties"]["plan"].keys():
        continue
    for cycle in polygon["properties"]["plan"][pType]:
        pointAnalysis = {
            "type": "FeatureCollection",
            "features": []
            }
        
        """ Get the orbit number """
        orbitNumber = eI.cyclePointOrbitNumber(cycle, point)
        
        """ Find the incidence angles """
        targetIncidenceStart = 22
        targetIncidenceEnd = 36
        dIncidence = 2.0
        guess = 1
        while guess != 0:
            spoint = eI.incidenceZeroDoppler(orbitNumber, point)
            guess = int((spoint["properties"]["incidence"]
                         - targetIncidenceStart)/dIncidence)
            orbitNumber += guess
            
        while spoint["properties"]["incidence"] < targetIncidenceEnd:
            pointAnalysis["features"].append(spoint)
            orbitNumber -= 1
            spoint = eI.incidenceZeroDoppler(orbitNumber, point)
        
        filename = "%s_%s.geojson" % (polygon["properties"]["ROI_No"], pType)
        with open(os.path.join(filepath, filename), "w") as f:
            f.write(json.dumps(pointAnalysis, indent=2))
    
# for k in range(20):
#     satPoint = incidenceZeroDoppler(sv, 
#                                     cidx,
#                                     k,
#                                     point)
    
#     pointAnalysis["features"].append(satPoint)


#%% Test
with open(os.path.join(filepath, "pointTStereo.geojson"), "w") as f:
    f.write(json.dumps(pointAnalysis, indent=2))
    
#%% Generate the incidence angle data. TIME INTESIVE
svIndex = idxarr[147]
rand_indexes = np.random.randint(0, cycle_jump[0]-1, 500)

for k, idx in enumerate(rand_indexes):
#for k, idx in enumerate(idxs):
    print(k)
    svIndex = idxarr[idx]    
    refSV = {"time": sv.measurementTime[svIndex].astype(str),
             "posvel": sv.measurementData[svIndex].tolist()}
    
    xG, xG_snorm, targetInc = groundPoint(sv, 
                                          svIndex, 
                                          targetIncidenceAngle = 30,
                                          plotSwaths = False)
    
    try:
        point = pointHistory(sv,
                             refSV,
                             xG, 
                             xG_snorm, 
                             targetInc,
                             tascarr)
        #_ = plotSpiderFigure(point)
        #_ = plotIncidenceFigure(point)
        
        #print(np.array(matchIncidence(point, threshold = 1.5)))
        
        writePointFile(filepath, point)
    except ValueError:
        print("Failed for svIndex %d" % svIndex)
        
#%% Analyze simulated data split
flist = glob(os.path.join(filepath, "*.json"))

#%% Analyze data from computed points
llh = []
cycle1OrbitNumber = []
repeatStereo = []
incidenceAngles = []
orbitNumbers = []
tests = {"points": []}
to_fix = []
to_from = []
for simfile in flist:
#for simfile in to_fix:
    with open(simfile, "r") as f:
        point = json.loads(f.read())
        
    #_ = plotIncidenceFigure(point)
    #_ = plotSpiderFigure(point)
    orbitNumA = point["target"]["satReference"]["orbitNumber"]
    fixPointTime(point, tascarr) # Needed to solve start of orbit problem at equator
    orbitNumB = point["target"]["satReference"]["orbitNumber"]
    if orbitNumB != orbitNumA:
        to_from.append((orbitNumB, orbitNumA))
        continue
    try:
        costs, incSets, orbNumbers, point = matchIncidencePermute(point,
                                                                  incidence_range = [22,32])
        llh.append(point["target"]["llh"])
        cycle1OrbitNumber.append(point["target"]["satReference"]["orbitNumber"])
        repeatStereo.append(costs)
        incidenceAngles.append(incSets)
        orbitNumbers.append(orbNumbers)
        point["solution"] = {"repeatStereo": costs.tolist(),
                             "incidenceAngles": incSets.tolist(),
                             "orbitNumbers": orbNumbers.tolist()}
        tests["points"].append(point)
        _ = writePointFile(filepath, point)
    except ValueError:
        to_fix.append(simfile)
        print("Please re-analyse: %s" % simfile)

#%%
# llh.append(point["target"]["llh"])
# cycle1OrbitNumber.append(point["target"]["satReference"]["orbitNumber"])
#counts.append(matchIncidence(point, threshold = 1.5))
lon = [l[1] for l in llh]
repeat = np.array([int(np.sum(d, axis=0)[0] > 0) for d in repeatStereo])
stereo = np.array([int(np.sum(d, axis=0)[1] > 0) for d in repeatStereo])
bth = repeat*stereo
print(np.argwhere(bth==0))

plt.figure()
plt.plot(lon, repeat, 'd', lon, stereo, 'o')
plt.grid()
plt.show()
plt.xlabel('longitude (deg)')
plt.ylabel('inidcator')
plt.legend(['repeat','stereo'])

plt.title('Possibility of repeat and stereo\nmeasurements near equator as a function of longitude\n orbit file: ET1 2031 N')

#%% New function to compute valid options
modified = []
to_remove = []
for simfile in to_fix:
    with open(simfile, "r") as f:
        point = json.loads(f.read())
    svIndex = idxarr[point["target"]["satReference"]["orbitNumber"]]    
    refSV = {"time": sv.measurementTime[svIndex].astype(str),
             "posvel": sv.measurementData[svIndex].tolist()}
    
    xG, xG_snorm, targetInc = groundPoint(sv, 
                                          svIndex, 
                                          targetIncidenceAngle = 30,
                                          plotSwaths = False)
    
    try:
        point = pointHistory(sv,
                             refSV,
                             xG, 
                             xG_snorm, 
                             targetInc,
                             tascarr)
        #_ = plotSpiderFigure(point)
        #_ = plotIncidenceFigure(point)
        
        #print(np.array(matchIncidence(point, threshold = 1.5)))
        
        simfile2 = writePointFile(filepath, point)
        if simfile2 != simfile:
            modified.append(simfile2)
            to_remove.append(simfile)
    except ValueError:
        print("Failed for svIndex %d" % svIndex)
   
#%%
with open(os.path.join(filepath, "analysis.json"), 'w') as f:
    f.write(json.dumps(tests, indent=2))