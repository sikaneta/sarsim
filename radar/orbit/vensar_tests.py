#%% -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:10:22 2022

@author: Ishuwa.Sikaneta
"""

from space.planets import venus
from orbit.pointing import simulation
from orbit.envision import loadSV

import numpy as np
import matplotlib.pyplot as plt
import json
import os


#%% Create a simulation object
eSim = simulation(planet = venus(),
                  e_ang = 14.28,
                  azAxis = 5.5,
                  elAxis = 0.6125,
                  carrier = 3.15e9)

#%% Load an oem orbit state vector file from Venus
sv = loadSV()

# with open(svfile, 'r') as f:
#     svecs = f.read().split('\n')
    
# svecs = [s.split() for s in svecs[16:-1]]
# svecs = [s for s in svecs if len(s)==7]
# svs = [(np.datetime64(s[0]), np.array([float(x) for x in s[1:]])*1e3) 
#        for s in svecs]

#%% Define sample simulation parameters
""" Select a subset of state vectors """
selection_range = [270, 480, 2]
mysvs = eSim.state(sv.measurementData, selection_range)
 
times = [(sv.measurementTime[k] - sv.measurementTime[0])/np.timedelta64(1, 's') 
         for k in range(*selection_range)]

xidx = 0
X = mysvs[1]

#%% Define the geometry
""" Define the off-nadir angle. Negative angles for right looking """
off_nadir = -18.7

""" Generate the covariances """  
covariances = {
    "spacecraft": {
        "description": "Errors in the orientation of the spacecraft.",
        "referenceVariables": "RollPitchYaw",
        "units": "radians",
        "R": ((np.diag([8.2e-3, 0.93e-3, 0.93e-3])/2)**2).tolist()
        },
    "instrument": {
        "description": """Errors in the pointing of the antenna. From JPL
                          spreadsheet cells 'SAR APE Pointing Budget'!D34:36
                          These are Allocation values and do not include a 20% 
                          margin.""",
        "referenceVariables": "AzimuthElevationTilt",
        "units": "radians",
        "R": ((np.diag([0.65e-3, 4.50e-3, 0.52e-3])/2)**2).tolist()
        },
    "orbitVelocity": {
        "description": "Errors in the orbit velocity vector",
        "referenceVariables": "VxVyVz",
        "units": "m/s",
        "R": (np.diag([0.2, 0.2, 0.2])**2).tolist()
        },
    "orbitAlongTrack": {
        "description": "Error in the orbit time (error in orbit angle)",
        "referenceVariables": "t",
        "units": "s",
        "R": (4.3/2)**2
        },
    "orbitAcrossTrack": {
        "description": "Error in the orbit across track position (orbit tube)",
        "referenceVariables": "dX",
        "units": "m",
        "R": np.diag([(600/2)**2, (850/2)**2])
        }
    }

#%%
# R_RPY = np.diag([4.8e-3, 0.4e-3, 1.1e-3])**2
# R_AEU = np.diag([])**2
# R_v = np.diag([0.2, 0.2, 0.2])**2
# R_t = 5**2
# R_p = 430**2
    
#%% Run the simulation on the defined values
res = eSim.simulateError(X, off_nadir, covariances, loglevel=3)

#%% Run the simulation over a range of different pitch and yaw variances
parray = np.arange(0.1, 0.4, 0.05)*1.0e-3
yarray = np.arange(0.1, 1.2, 0.05)*1.0e-3

res_array = []
for p in parray:
    for y in yarray:
        print("Pitch: %0.2e, Yaw: %0.2e" % (p,y))
        covariances["spacecraft"]["R"] = np.diag([5.5e-3, p, y])**2
        dd = eSim.simulateError(X, off_nadir, covariances) 
        res_array.append(dd)

#%% Read the Doppler error rate from the result of the previous cell
dprate = np.array([r["ErrorRate"]["Doppler"] 
                   for r in res_array]).reshape((len(parray), len(yarray)))

#%% Make a contour plot of the percentage of Doppler errors
plt.figure()
plt.contour(yarray*1e3, parray*1e3, dprate, np.arange(1,10,0.5), origin='lower')
plt.xlabel("yaw (mrad)")
plt.ylabel("pitch (mrad)")
plt.colorbar()
plt.grid()
plt.contour(yarray*1e3, 
            parray*1e3, 
            dprate, 
            np.array([5]), 
            origin='lower', 
            colors = ['red'])
plt.title("Percentage of R-MIS-PER-1045 violations (Doppler)")
plt.show()

#%% Save the results of scanning over pitch and yaw to file
filepath = os.path.join(r"C:\Users",
                        r"ishuwa.sikaneta",
                        r"OneDrive - ESA",
                        r"Documents",
                        r"ESTEC",
                        r"Envision",
                        r"PointingSimulations")

#%%
filename = r"pitchRoll49PitchYawRelation.json"
with open(os.path.join(filepath, filename), "w") as f:
    f.write(json.dumps(res_array, indent=2))

#%% Run a simulation to calculate error rates for different points in the orbit
#R_RPY = np.diag([4.8e-3, 0.4e-3, 1.1e-3])**2
res2 = []
for k in range(len(mysvs)):
    X = mysvs[k]
    dd = eSim.simulateError(X, 
                            off_nadir, 
                            covariances)
                            # R_RPY = R_RPY, 
                            # R_v = R_v, 
                            # R_t = R_t, 
                            # R_p = R_p)
    res2.append(dd)
    print("%0.2f" % (float(k+1)/len(mysvs)))


#%% Save the generated simulation results to file
#filename = r"sim-%s.json" % dt.now().strftime("%Y-%m-%dT%H%M%S")
with open(os.path.join(filepath, filename), "w") as f:
    f.write(json.dumps(res2, indent=2))

#%% Reload the file and plot the some results
filename = r"incidence20-2023-02-07T121846.json"
with open(os.path.join(filepath, filename), "r") as f:
    res18p7 = json.loads(f.read())

    
#%%    
dErrorRate = np.array([r["ErrorRate"]["Doppler"] for r in res2])
sErrorRate = np.array([r["ErrorRate"]["Swath"] for r in res2])
plt.figure()
plt.plot(times, dErrorRate)
plt.plot(times, sErrorRate)
plt.ylim(0,5)
plt.title("20 degree incidence angle")
plt.ylabel("Percentage in excess of threshold")
plt.xlabel("State Vector %s +Time (s)" % np.datetime_as_string(sv.measurementTime[0]))
plt.legend(["Doppler", "Swath"])
plt.grid()
plt.show()
print(np.mean(sErrorRate))
print(np.var(sErrorRate))
print(np.var(dErrorRate))
print(np.mean(dErrorRate))


#%% Read the data
""" Save data to file """
filename = r"pitchYawRoll5p5-%s.json" % "2023-02-09T200920"
with open(os.path.join(filepath, filename), "r") as f:
    res5p5 = json.loads(f.read())
    
filename = r"pitchYawRoll4p5-%s.json" % "2023-02-09T200920"
with open(os.path.join(filepath, filename), "r") as f:
    res4p5 = json.loads(f.read())
    
#%% Make plot
parray = np.arange(0.3, 0.6, 0.05)*1.0e-3
yarray = np.arange(0.3, 1.4, 0.05)*1.0e-3

#%% 4p5
dprate = np.array([r["ErrorRate"]["Doppler"]
            for r in res4p5]).reshape((len(parray), len(yarray)))

#%%
plt.figure()
plt.contour(yarray*1e3, parray*1e3, dprate, np.arange(1,10,1), origin='lower')
plt.xlabel("yaw (mrad)")
plt.ylabel("pitch (mrad)")
plt.colorbar()
plt.grid()
plt.contour(yarray*1e3,
            parray*1e3,
            dprate,
            np.array([5]),
            origin='lower',
            colors = ['red'])
plt.title("Percentage of R-MIS-PER-1045 violations (Doppler)")
plt.show()

#%% Read the Doppler error rate from the result of the previous cell
dprate = np.array([r["ErrorRate"]["Doppler"]
            for r in res5p5]).reshape((len(parray), len(yarray)))

#%% Make a contour plot of the percentage of Doppler errors
plt.figure()
plt.contour(yarray*1e3, parray*1e3, dprate, np.arange(1,10,1), origin='lower')
plt.xlabel("yaw (mrad)")
plt.ylabel("pitch (mrad)")
plt.colorbar()
plt.grid()
plt.contour(yarray*1e3,
            parray*1e3,
            dprate,
            np.array([5]),
            origin='lower',
            colors = ['red'])
plt.title("Percentage of R-MIS-PER-1045 violations (Doppler)")
plt.show()