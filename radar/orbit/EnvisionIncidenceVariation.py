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
    orbitNumber = [oN if t > timeArray[oN] else oN-1 
                   for oN, t in zip(nearestOrbit,times)]
    return orbitNumber

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
        
        """ Check that we're at the right solution """
        if np.abs(inc - targetInc) > 30:
            eta += np.timedelta64(int(venSAR.period/2), 's')
            rvec, inc, satSV  = computeImagingGeometry(sv, eta, xG, xG_snormal)
        
        """ The next two lines are to reset the period """
        myOrbit = orbit(planet=sv.planet, angleUnits="radians")
        orbitAngle, ascendingNode = myOrbit.setFromStateVector(satSV[1])
        period = np.timedelta64(int(myOrbit.period*1e6),'us')
        
        """ Generate an array of times to examine """
        etarange = satSV[0] + orbitRange*period
        
        """ Compute the imaging geometry to broadside at these times """
        try:
            options = [computeImagingGeometry(sv, eta, xG, xG_snormal) 
                       for eta in etarange]
        except IndexError:
            print("Faile for time: %s" % eta.astype(str))
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
    with open(os.path.join(filepath, filename), 'w') as f:
        f.write(json.dumps(point, indent = 2))
        
# Function to find the number of cases where all cycles work
def matchIncidence(point, threshold = 1.5):
    oNS = np.arange(5)
    # groundPoint = tuple(point["target"]["llh"])
    myincidences = [[p["incidence"]
                     for p in cycle] 
                    for cycle in point["cycle"]]
    
    myorbitNum = [[p["orbitNumber"]%5 for p in cycle] 
                  for cycle in point["cycle"]]
    
    #refOrbitNumber = point["target"]["satReference"]["orbitNumber"]
    targetInc = point["target"]["satReference"]["incidence"]
    
    pl = [[[x for x,k in zip(myinc, myorb) if k == n] 
            for n in oNS] for myinc, myorb in zip(myincidences, myorbitNum)]
    
    matches = [[len([l for l in p if np.abs(l-targetInc) < threshold]) 
                for p in ple] 
               for ple in pl]

    return matches
 
#%%#%% Generate some data The orbit number is the actually the state vector number
# svindeces = [29803]
# svindeces = [1507] # equator
# svindeces = [idxarr[3443]]
if 'linux' in sys.platform:
    filepath = '/users/isikanet/local/data/cycles'
else:
    filepath = r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\Envision\PointingSimulations\cycles"

#%%
svIndex = idxarr[147]
rand_indexes = np.random.randint(0, cycle_jump[0]-1, 500)

for k, idx in enumerate(rand_indexes):
    print(k)
    svIndex = idxarr[idx]    
    refSV = {"time": sv.measurementTime[svIndex].astype(str),
             "posvel": sv.measurementData[svIndex].tolist()}
    
    xG, xG_snorm, targetInc = groundPoint(sv, svIndex, targetIncidenceAngle = 30)
    
    try:
        point = pointHistory(sv,
                             refSV,
                             xG, 
                             xG_snorm, 
                             targetInc,
                             tascarr)
        #_ = plotSpiderFigure(point)
        
        #print(np.array(matchIncidence(point, threshold = 1.5)))
        
        writePointFile(filepath, point)
    except ValueError:
        print("Failed for svIndex %d" % svIndex)
    
    
#%% Analyze simulated data
flist = glob(os.path.join(filepath, "*.json"))

counts = []

for simfile in flist:
    with open(simfile, "r") as f:
        point = json.loads(f.read())
    counts.append(matchIncidence(point, threshold = 1.5))
    # dd = np.sum(matchIncidence(point, threshold = 1.5), axis=0)
    # if np.any(dd == 0):
    #     fail_list.append((simfile, len(np.where(dd > 0)[0])))
    
dist = np.array([len(np.where(np.sum(x, axis=0)>0)[0]) for x in counts])