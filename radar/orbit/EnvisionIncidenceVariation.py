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

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from glob import glob
import sys

#%% Load state vectors for envision
sv = loadSV(toPCR = False)

#%% Get the period
venSAR = orbit(planet=sv.planet, angleUnits="radians")

#%% Estimate ascending Node times
idx = 0
N = len(sv.measurementTime)
tascarr = []
idxarr = []

""" Iterate and find all asceding node crossing times """
srange = np.arange(0)
while idx < N:
    orbitAngle, ascendingNode = venSAR.setFromStateVector(sv.measurementData[idx])
    dtasc = venSAR.period + venSAR.computeT(0)-venSAR.computeT(orbitAngle)
    tasc = sv.measurementTime[idx] + np.timedelta64(int(dtasc), 's')
    tascarr.append(tasc)
    try:
        idx = findNearest(sv.measurementTime, tasc)
        idxarr.append(idx)
        if idx%1000 == 1:
            print(idx)
    except IndexError:
        idx = N + 1

#%% Check the time differences
if 'linux' not in sys.platform:
    kep = [venSAR.state2kepler(sv.measurementData[k]) for k in idxarr]
    z = [sv.measurementData[k][2] for k in idxarr]
    plt.figure()
    plt.plot(tascarr[0:-1], z)
    plt.grid()
    plt.show()
    
    period = [2*np.pi*np.sqrt(venSAR.state2kepler(sv.measurementData[k])['a']**3
                              /venSAR.planet.GM) for k in idxarr]
    plt.figure()
    plt.plot(tascarr[0:-1], period)
    plt.xlabel('Time')
    plt.ylabel('Orbit Period (s)')
    plt.title('Orbit period for ET1 2031 N')
    plt.grid()
    plt.show()
    
    
    a = [venSAR.state2kepler(sv.measurementData[k])['a']/1e3 for k in idxarr]
    plt.figure()
    plt.plot(tascarr[0:-1], a)
    plt.xlabel('Time')
    plt.ylabel('Orbit semi-major axis (km)')
    plt.title('Orbit semi-major axis for ET1 2031 N')
    plt.grid()
    plt.show()
    
    plt.figure()
    plt.plot(tascarr[0:-1], np.diff(tascarr)/np.timedelta64(1,'m'))
    plt.grid()
    plt.title('Orbit period (minutes)')
    plt.xlabel('Time')
    plt.ylabel('Orbit period (minutes)')
    plt.show()
    
    perigee = [k["perigee"] for k in kep]
    plt.figure()
    plt.plot(tascarr[0:-1], np.degrees(np.unwrap(perigee)))
    plt.xlabel('Time')
    plt.ylabel('Orbit perigee angle (deg)')
    plt.title('Orbit perigee angle for ET1 2031 N')
    plt.grid()
    plt.show()

#%%
cycle_jump = np.argwhere(np.diff(np.diff(tascarr)/np.timedelta64(1,'m')) > 0.02)[:,0] + 1


""" If everything checks out, then we have an array of 
    times of the ascending node. This will help to define
    the orbit numbers for our calculations. Values are in
    the array tascarr """

#%% Reload state vectors but in VCR frame
sv = loadSV(toPCR = True)
# nsvindecesPerCycle = 175217
# cycleDuration = np.timedelta64(14600, 'h')
# cycleDuration = np.timedelta64(350412, 'm')

#%% Function to find the orbit number
def getOrbitNumber(timeArray, times):
    nearestOrbit = [findNearest(timeArray, t) for t in times]
    # orbitNumber = [oN if t > timeArray[oN] else oN-1 
    #                for oN, t in zip(nearestOrbit,times)]
    # return orbitNumber
    return nearestOrbit

def fixPointTime(point, tascarr):
    oN = findNearest(tascarr, np.datetime64(point["target"]["satReference"]["stateVector"]["time"]))
    point["target"]["satReference"]["orbitNumber"] = oN
    for cycle in point["cycle"]:
        for pt in cycle:
            pt["orbitNumber"] = findNearest(tascarr, np.datetime64(pt["state_vector"]["time"]))
            
        
def testBroadside(pts, xG):
    svel = np.array([p["state_vector"]["satvel"] for p in pts])
    rvec = np.array([np.array(p["state_vector"]["satpos"]) - xG for p in pts])
    svel = np.diag(1/np.linalg.norm(svel, axis=1)).dot(svel)
    rvec = np.diag(1/np.linalg.norm(rvec, axis=1)).dot(rvec)
    
    deltaT = np.array([np.datetime64(p["state_vector"]["time"])
                       -np.datetime64(pts[0]["state_vector"]["time"]) for p in pts])
    deltaT = deltaT/np.timedelta64(1,'s')
    dotprod = np.sum(svel*rvec, axis=1)
    return dotprod, deltaT

# This should probably be changed to allow for reference cycle other than 1
def cycleIndex(sv, 
               targetSV, 
               dv_threshold = 500,
               ddv_threshold = 6000,
               dx_threshold = 1.0e6):
    dd = np.array([sv - np.array(targetSV) for sv in sv.measurementData])
    dp = np.linalg.norm(dd[:,:3], axis=1)
    dv = np.linalg.norm(dd[:,3:], axis=1)
    
    # Find the minima for repeated satellite positions
    qq = np.argwhere(np.array(dv) < dv_threshold)
    qq_diff = np.diff(qq, axis=0)
    qqi = np.argwhere(np.abs(qq_diff) > ddv_threshold)[:,0]
    qqi = [0] + list(qqi) + [len(qq_diff)-1] 
    
    sets = [(qq[k+1][0],qq[kk][0]) for k,kk in zip(qqi[:-1], qqi[1:])]
    cycle_idxs = [s[0] + np.argmin(dp[s[0]:s[1]]) for s in sets]
    return [cyc for cyc in cycle_idxs if dp[cyc] < dx_threshold]

# Compute the history of imaging of a particular point
def pointHistory(sv, 
                 refSV,
                 xG, 
                 xG_snormal, 
                 targetInc,
                 tascarr,
                 cycle_threshold = np.timedelta64(90,'D'),
                 orbitRange = np.arange(-20,21)):
    
    
    # Compute seed for satellite position for subsequenct cycles
    cycle_idxs = cycleIndex(sv, refSV["posvel"])
    time_points = [sv.measurementTime[k] for k in cycle_idxs]
    time_points = [t for t in time_points 
                   if t - np.datetime64(refSV["time"]) > cycle_threshold]
    
    # Compute the orbit number on reference cycle
    refOrbitNumber = getOrbitNumber(tascarr, 
                                    [np.datetime64(refSV["time"])])[0]
    
    point = {"target": {"xyz": xG.tolist(), 
                        "llh": list(sv.xyz2polar(xG)),
                        "normal": xG_snormal.tolist(),
                        "satReference": {"stateVector": refSV,
                                         "orbitNumber": refOrbitNumber,
                                         "incidence": targetInc}
                        },
             "processingTimestamp": np.datetime64("now").astype(str),
             "cycle": []
             }
    
    for eta in tqdm(time_points):
        """ Compute the incidence angle at the cycle. If this angle is
            greater than 60 degrees (i.e. wrong solution), then flip over half 
            an orbit """
        rvec, inc, satSV  = computeImagingGeometry(sv, eta, xG, xG_snormal)
        
        """ The next two lines are to reset the period """
        myOrbit = orbit(planet=sv.planet, angleUnits="radians")
        orbitAngle, ascendingNode = myOrbit.setFromStateVector(satSV[1])
        period = np.timedelta64(int(myOrbit.period*1e6),'us')
        
        """ Check that we're at the right solution """
        if np.abs(inc - targetInc) > 90:
            eta += period/2
            rvec, inc, satSV  = computeImagingGeometry(sv, eta, xG, xG_snormal)
        
        """ Generate an array of times to examine """
        etarange = satSV[0] + orbitRange*period
        
        """ Compute the imaging geometry to broadside at these times """
        try:
            options = [computeImagingGeometry(sv, eta, xG, xG_snormal) 
                       for eta in etarange]
        except IndexError:
            print("Failed for time: %s" % eta.astype(str))
            continue
        
        """ Compute the orbit number """
        orbitNumber = getOrbitNumber(tascarr, 
                                     [o[2][0] for o in options])
        
        """ Write to json/dict """
        point["cycle"].append(
                [
                 {
                  "incidence": o[1],
                  "range": np.linalg.norm(o[0]),
                  "orbitNumber": oN,
                  "state_vector": 
                      {
                       "time": np.datetime_as_string(o[2][0]),
                       "satpos": o[2][1][:3].tolist(),
                       "satvel": o[2][1][3:].tolist()
                      },
                  "llh": list(sv.xyz2polar(o[2][1][:3]))
                 } for o, oN in zip(options, orbitNumber)
                ]
            )
    return point

def groundPoint(sv, 
                svIndex, 
                targetIncidenceAngle = 30, 
                groundSwath = 57e3,
                maxOffNadir = 40,
                offNadirStep = 0.01,
                plotSwaths = False):

    svindeces = [svIndex]
    off_nadir = np.sign(targetIncidenceAngle)*np.arange(0,maxOffNadir,offNadirStep)
    incidence = np.zeros((len(off_nadir), len(svindeces)), dtype = float)
    ranges = np.zeros((len(off_nadir), len(svindeces)), dtype = float)
    rhats = np.zeros((len(off_nadir), 3, len(svindeces)), dtype = float)
    ground_range = np.zeros((len(off_nadir), len(svindeces)), dtype = float)
    
    # Use the getTiming function to compute the geometry
    for k, idx in enumerate(svindeces):
        (ranges[:,k], 
         rhats[:,:,k], 
         incidence[:,k], 
         _, 
         ground_range[:,k]) = getTiming(sv, 
                        np.radians(off_nadir), 
                        idx = idx)
                                        
    targetIDX = np.argmin(np.abs(incidence-targetIncidenceAngle))
    targetInc = incidence[targetIDX][0]
    idx = np.argwhere(np.abs(ground_range - 
                             ground_range[targetIDX]) 
                      < groundSwath/2)[:,0]
    
    # Compute the coordinates of the ground point and surface normal
    xG = sv.measurementData[svindeces[0]][0:3] + ranges[targetIDX,0]*rhats[targetIDX,:,0]
    xG_snormal = surfaceNormal(xG,sv)
    groundPoint = tuple(sv.xyz2polar(xG))
    
    
    # Compute the satellite position 5 orbits later
    targetSV = sv.measurementData[svindeces[0]]
    myOrbit = orbit(planet=sv.planet, angleUnits="radians")
    orbitAngle, ascendingNode = myOrbit.setFromStateVector(targetSV)
    period = np.timedelta64(int(myOrbit.period), 's')
    targetSVtime = sv.measurementTime[svindeces[0]] + 5*period
    R, inc, satSV = computeImagingGeometry(sv, targetSVtime, xG, xG_snormal)
    sposDiff = np.linalg.norm(targetSV[:3] - satSV[1][:3])
    
    #
    if(plotSwaths):
        plt.figure()
        plt.plot(ground_range/1e3, incidence)
        plt.plot(ground_range[idx]/1e3, incidence[idx], 'g.')
        plt.plot(ground_range[targetIDX]/1e3, incidence[targetIDX], 'ro')
        grdiff = np.linalg.norm(sposDiff)/1e3
        plt.plot(ground_range/1e3 - grdiff, incidence)
        plt.plot(ground_range[idx]/1e3 - grdiff, incidence[idx], 'g.')
        plt.plot(ground_range[targetIDX]/1e3 - grdiff, incidence[targetIDX], 'ro')
        plt.xlabel("ground range (km)")
        plt.ylabel("Incidence angle (deg)")
        plt.grid()
        plt.title('Ground point: lat: %0.2f (deg), lon %0.2f (deg), hae: %0.2f (m)\nFive orbit difference' % groundPoint)
        plt.show()    

    return xG, xG_snormal, targetInc

def plotIncidenceFigure(point):
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

def plotSpiderFigure(point):
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

def writePointFile(filepath, point):
    orbitNumber = point["target"]["satReference"]["orbitNumber"]
    incidence = point["target"]["satReference"]["incidence"]
    sgn = "m" if incidence < 0 else "p"
    filename = "analysis_%0.4d_%s%6.4f.json" % (orbitNumber, sgn, np.abs(incidence))
    targetFile = os.path.join(filepath, filename)
    with open(os.path.join(filepath, filename), 'w') as f:
        f.write(json.dumps(point, indent = 2))
        
    return targetFile
        
# Function to find the number of cases where all cycles work
def matchIncidence(point, threshold = 1.5):
    oNS = np.arange(5)
    # groundPoint = tuple(point["target"]["llh"])
    myincidences = [[p["incidence"]
                     for p in cycle] 
                    for cycle in point["cycle"]]
    
    myorbitNum = [[p["orbitNumber"] for p in cycle] 
                  for cycle in point["cycle"]]
    
    #refOrbitNumber = point["target"]["satReference"]["orbitNumber"]
    targetInc = point["target"]["satReference"]["incidence"]
    
    pl = [[[(x,k) for x,k in zip(myinc, myorb) if k%5 == n] 
            for n in oNS] for myinc, myorb in zip(myincidences, myorbitNum)]
    
    candidates = [[[l for l in p if np.abs(l[0]-targetInc) < threshold] 
                   for p in ple] for ple in pl]
    
    matches = [[len(p) for p in ple] for ple in candidates]

    return matches, candidates

def toMatrix(incidences, initOEN, fill = np.nan):
    incidences = [np.nan for k in range(0,initOEN)] + incidences
    append = range(len(incidences)%5,5) if len(incidences)%5 > 0 else []
    incidences += [np.nan for k in append]
    rows = len(incidences)//5
    return np.array(incidences).reshape((rows,5)).T

# Function to find the number of cases where all cycles work
def bestValue(mtx, offset, index, threshold):
    idx = np.nanargmin(np.abs(mtx - offset[0]), axis=1)[index]
    cost = np.abs(mtx[index, idx] - offset[0])/threshold
    m,n = mtx.shape
    orbitNumber = n*index + idx
    return mtx[index, idx], idx, cost, orbitNumber, offset[1]

def bestIncidence(mtx, offset, index, threshold):
    idx = np.nanargmin(np.abs(mtx - offset), axis=1)[index]
    m,n = mtx.shape
    orbitNumber = n*index + idx
    return mtx[index, idx], idx, orbitNumber

def pathRS(cr, cs, idx):
    r = len(list(set([(x[0], x[2]) for x in cr if x[1] == idx and x[2] < 1])))
    s = len(list(set([(x[0], x[2]) for x in cs if x[1] == idx and x[2] < 1])))
    
    return np.array([r,s])

def matchIncidenceOp(point, rThreshold = 1.5, sOffset = 5.5, sThreshold = 1.5):
    # oNS = np.arange(5)
    
    c1inc = point["target"]["satReference"]["incidence"]
    c1ON = point["target"]["satReference"]["orbitNumber"]
    c1OEN = c1ON%5
    
    myincidences = [[p["incidence"] for p in c] for c in point["cycle"]]
    myorbitNum = [[p["orbitNumber"] for p in c] for c in point["cycle"]]
    [cyc2, cyc3, cyc4, cyc5, cyc6] = [toMatrix(inc, num[0]%5) for inc, num 
                                      in zip(myincidences, myorbitNum)]
    
    dataOrbitNumber = [m[0] - m[0]%5 for m in myorbitNum]
    
    allCycles = [cyc2, cyc3, cyc4, cyc5, cyc6]
    
    def prev(cyc, prev_res, offset, threshold):
        offset = [i[0] + offset for i in prev_res]
        return [bestIncidence(cyc, o, c1OEN, threshold) for o in offset]
    
    c1 = [(c1inc, c1OEN, None)]
        
    c2 = prev(cyc2, c1, sOffset, sThreshold)
    
    c3 = list(set(prev(cyc3, c1, 0, rThreshold) + 
                  prev(cyc3, c2, -sOffset, sThreshold)))
    
    c4 = list(set(prev(cyc4, c1, sOffset, sThreshold) + 
                  prev(cyc4, c2, 0, rThreshold) +
                  prev(cyc4, c3, sOffset, sThreshold)))
    
    c5 = list(set(prev(cyc5, c1, 0, rThreshold) +
                  prev(cyc5, c2, -sOffset, sThreshold) + 
                  prev(cyc5, c3, 0, rThreshold) +
                  prev(cyc5, c4, -sOffset, sThreshold)))
    
    c6 = list(set(prev(cyc6, c1, sOffset, sThreshold) + 
                  prev(cyc6, c2, 0, rThreshold) +
                  prev(cyc6, c3, sOffset, sThreshold) + 
                  prev(cyc6, c4, 0, rThreshold) +
                  prev(cyc6, c5, sOffset, sThreshold)))
    
    paths = [[c2[0][1], w[1], x[1], y[1], z[1]] 
             for w in c3 
             for x in c4 
             for y in c5 
             for z in c6]
    
    orbNumbers = np.array([[dON + cyc.shape[0]*p + c1OEN 
                            for p, cyc, dON in zip(path, 
                                                   allCycles, 
                                                   dataOrbitNumber)] 
                            for path in  paths])
    
    angles = np.array([[c1inc, c2[0][0], w[0], x[0], y[0], z[0]] 
                       for w in c3 
                       for x in c4 
                       for y in c5 
                       for z in c6])
    
    sVct = np.array([0,1,0,1,0,1])*sOffset
    rThrVct = np.array([1,0,1,0,1,0])*rThreshold
    sThrVct = np.array([0,1,0,1,0,1])*sThreshold
    
    sMtx = np.array([[0,1,0,1,0,1],
                     [0,0,1,0,1,0],
                     [0,0,0,1,0,1],
                     [0,0,0,0,1,0],
                     [0,0,0,0,0,1],
                     [0,0,0,0,0,0]])
                     
    rMtx = np.array([[0,0,1,0,1,0],
                     [0,0,0,1,0,1],
                     [0,0,0,0,1,0],
                     [0,0,0,0,0,1],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0]])
    
    costs = []
    for ang in angles:
        x = (ang - sVct)/(rThrVct + sThrVct)
        y = np.abs(np.tile(x, 6).reshape((6,6)) - 
                   np.tile(x, 6).reshape((6,6)).T)
        
        rVals = y*rMtx
        sVals = y*sMtx
        costs.append([np.sum((rVals>0).astype(int) * (rVals<1).astype(int)),
              np.sum((sVals>0).astype(int) * (sVals<1).astype(int))])
    
    return np.array(costs), angles[:,1:], orbNumbers
    
def matchIncidencePermute(point, 
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
            additional_orbits = extend(point, 
                                       sv, 
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
   
def extend(point, 
           sv,
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
    myOrbit = orbit(planet=sv.planet, angleUnits="radians")
    orbitAngle, ascendingNode = myOrbit.setFromStateVector(refSV)
    period = np.timedelta64(int(myOrbit.period*1e6),'us')
    
    """ Generate an array of times to examine """
    etarange = refTime + orbitRange*period
    
    """ Compute the imaging geometry to broadside at these times """
    try:
        options = [computeImagingGeometry(sv, eta, xG, xG_snormal) 
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
              "llh": list(sv.xyz2polar(o[2][1][:3]))
             } for o, oN in zip(options, orbitNumber)
            ]
    
    return add_orbits
 
#%%#%% Generate some data The orbit number is the actually the state vector number
# svindeces = [29803]
# svindeces = [1507] # equator
# svindeces = [idxarr[3443]]
if 'linux' in sys.platform:
    filepath = '/users/isikanet/local/data/cycles'
else:
    filepath = r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\Envision\PointingSimulations\cycles"

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
        costs, incSets, orbNumbers, point = matchIncidencePermute(point)
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