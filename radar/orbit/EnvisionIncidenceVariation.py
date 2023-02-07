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

cycle_jump = np.argwhere(np.diff(np.diff(tascarr)/np.timedelta64(1,'m')) > 0.02)[:,0] + 1


""" If everything checks out, then we have an array of 
    times of the ascending node. This will help to define
    the orbit numbers for our calculations. Values are in
    the array tascarr """

#%% Reload state vectors but in VCR frame
sv = loadSV(toPCR = True)
nsvindecesPerCycle = 175217
cycleDuration = np.timedelta64(14600, 'h')
cycleDuration = np.timedelta64(350412, 'm')

#%% Generate some data The orbit number is the actually the state vector number
svindeces = [29803]
svindeces = [9903]
off_nadir = np.arange(00,40,0.01)
incidence = np.zeros((len(off_nadir), len(svindeces)), dtype = float)
ranges = np.zeros((len(off_nadir), len(svindeces)), dtype = float)
rhats = np.zeros((len(off_nadir), 3, len(svindeces)), dtype = float)
ground_range = np.zeros((len(off_nadir), len(svindeces)), dtype = float)

#%%
for k, idx in enumerate(svindeces):
    (ranges[:,k], 
     rhats[:,:,k], 
     incidence[:,k], 
     _, 
     ground_range[:,k]) = getTiming(sv, 
                    np.radians(off_nadir), 
                    idx = idx)

#%% Plot the data
plt.figure()
plt.plot(off_nadir, incidence)
plt.xlabel("Off-nadir angle (deg)")
plt.ylabel("Incidence angle (deg)")
plt.grid()
plt.show()

targetIncidenceAngle = 30
groundSwath = 57e3
targetIDX = np.argmin(np.abs(incidence-targetIncidenceAngle))
idx = np.argwhere(np.abs(ground_range - 
                         ground_range[targetIDX]) 
                  < groundSwath/2)[:,0]

#%% Compute the coordinates of the ground point and surface normal
xG = sv.measurementData[svindeces[0]][0:3] + ranges[targetIDX,0]*rhats[targetIDX,:,0]
xG_snormal = surfaceNormal(xG,sv)
groundPoint = tuple(sv.xyz2polar(xG))


#%% Compute the satellite position on the next orbit
targetSV = sv.measurementData[svindeces[0]]
venPeriod = np.timedelta64(int(venSAR.period), 's')
targetSVtime = sv.measurementTime[svindeces[0]] + 5*venPeriod
R, inc, satSV = computeImagingGeometry(sv, targetSVtime, xG, xG_snormal)
sposDiff = np.linalg.norm(targetSV[:3] - satSV[1][:3])
print(np.linalg.norm(sposDiff))

#%%
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
plt.title('Ground point: lat: %0.2f (deg), lon %0.2f (deg), hae: %0.2f (m)' % groundPoint)
plt.show()

#%% Compute time between cycles
cycle_durations = np.diff([tascarr[0]] + [tascarr[k] 
                        for k in cycle_jump])
# cycle_durations = [np.timedelta64(350729,'m'),
#                    np.timedelta64(350729,'m'),
#                    np.timedelta64(350729,'m'),
#                    np.timedelta64(350487,'m'),
#                    np.timedelta64(350487,'m')
#                   ]

#%% Function to find the orbit number
def getOrbitNumber(timeArray, times):
    nearestOrbit = [findNearest(timeArray, t) for t in times]
    orbitNumber = [oN if t > timeArray[oN] else oN-1 
                   for oN, t in zip(nearestOrbit,times)]
    return orbitNumber

#%% Main work cell. Compute incidence angles across cycles
time_points = sv.measurementTime[svindeces[0]] + np.cumsum(cycle_durations)
orbitRange = np.arange(-10,11)
point = {"target": {"xyz": xG.tolist(), 
                    "llh": list(sv.xyz2polar(xG))
                    },
         "cycle": []
         }

for eta in tqdm(time_points):
    """ Compute the incidence angle at the cycle. If this angle is
        greater than 60 degrees (i.e. wrong solution), then flip over half 
        an orbit """
    rvec, inc, satSV  = computeImagingGeometry(sv, eta, xG, xG_snormal)
    
    """ Check that we're at the right solution """
    if np.abs(inc) > 60:
        eta += np.timedelta64(int(venSAR.period/2), 's')
        rvec, inc, satSV  = computeImagingGeometry(sv, eta, xG, xG_snormal)
    
    """ The next two lines are to reset the period """
    orbitAngle, ascendingNode = venSAR.setFromStateVector(satSV[1])
    period = np.timedelta64(int(venSAR.period*1e6),'us')
    
    """ Generate an array of times to examine """
    etarange = satSV[0] + orbitRange*period
    
    """ Compute the imaging geometry to broadside at these times """
    options = [computeImagingGeometry(sv, eta, xG, xG_snormal) 
               for eta in etarange]
    
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

#%%
symbols = ['.', 'o', 'x', 'd', 's']
colors = ['red', 'gold', 'slateblue', 'darkviolet', 'limegreen']
plt.figure()
myincidences = [[p["incidence"] 
                 for p in cycle] 
                for cycle in point["cycle"]]

myorbitNum = [[p["orbitNumber"]%5 for p in cycle] 
              for cycle in point["cycle"]]

for data,symbol in zip(myincidences, symbols):
    plt.plot(orbitRange, data, symbol)
    
plt.grid()
x_axis = np.arange(len(myincidences[0]))
y_axis = np.ones_like(x_axis)*incidence[targetIDX]
plt.plot(orbitRange, y_axis)
plt.xlabel('orbit')
plt.ylabel('Incidence Angle (deg)')
plt.legend(['cycle 2', 'cycle 3', 'cycle 4', 'cycle 5', 'cycle 6', 'cycle 1'])
plt.title('Incidence to point lat: %0.2f (deg), lon %0.2f (deg), hae: %0.2f (m)' % groundPoint)
plt.show()

targetInc = incidence[targetIDX][0]
#%%
oNS = np.arange(5)
pl = [[[x for x,k in zip(myinc, myorb) if k == n] 
        for n in oNS] for myinc, myorb in zip(myincidences,myorbitNum)]

#%% Compute the orbit number on cycle 1
cycle1orbitNumber = getOrbitNumber(tascarr, 
                                   [sv.measurementTime[svindeces[0]]])[0]
cycle1orbitNumber = cycle1orbitNumber%5 + 1

#%% Plot the incidence differences
for cycleNum in range(5):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    for k in range(5):
        r = np.array(pl[cycleNum][k]) - targetInc
        th = np.angle(np.exp(k*1j*np.pi/5)*r)
        ax.plot(th, np.abs(r), 'o')
    
    ax.set_rticks([1.5, 5, 10]) 
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
    plt.title("Cycle %d" % (cycleNum + 2))
    plt.show()
