#%% -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:10:22 2022

@author: Ishuwa.Sikaneta
"""

from space.planets import venus
from orbit.pointing import simulation

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime as dt
import os


#%% Create a simulation object
eSim = simulation(planet = venus(),
                  e_ang = 14.28,
                  azAxis = 6.0,
                  elAxis = 0.6,
                  carrier = 3.15e9)

#%% Load an oem orbit state vector file from Venus
svfile = os.path.join(r"C:\Users",
                          r"ishuwa.sikaneta",
                          r"OneDrive - ESA",
                          r"Documents",
                          r"ESTEC",
                          r"Envision",
                          r"Orbits",
                          r"EnVision_ALT_T4_2032_SouthVOI.oem")

with open(svfile, 'r') as f:
    svecs = f.read().split('\n')
    
svecs = [s.split() for s in svecs[16:-1]]
svecs = [s for s in svecs if len(s)==7]
svs = [(np.datetime64(s[0]), np.array([float(x) for x in s[1:]])*1e3) 
       for s in svecs]

#%% Define sample simulation parameters
""" First get the state vector """
mysvs = eSim.state(svs, [270, 480, 10])
xidx = 0
X = mysvs[xidx][1]

""" Define the off-nadir angle. Negative angles for right looking """
off_nadir = -30

""" Generate the covariances """  
R_RPY = np.diag([4.8e-3, 0.4e-3, 1.1e-3])**2
R_v = np.diag([0.2, 0.2, 0.2])**2
R_t = 5**2
R_p = 430**2
    
#%% Run the simulation on the defined values
eSim.simulateError(X, 
                   off_nadir, 
                   R_RPY = R_RPY, 
                   R_v = R_v, 
                   R_t = R_t, 
                   R_p = R_p,
                   loglevel=3)

#%% Run the simulation over a range of different pitch and yaw variances
parray = np.arange(0.1, 0.6, 0.05)*1.0e-3
yarray = np.arange(0.1, 2.0, 0.1)*1.0e-3

res_array = []
for p in parray:
    for y in yarray:
        print("Pitch: %0.2e, Yaw: %0.2e" % (p,y))
        R_RPY = np.diag([4.9e-3, p, y])**2
        dd = eSim.simulateError(X, 
                                off_nadir, 
                                R_RPY = R_RPY, 
                                R_v = R_v, 
                                R_t = R_t, 
                                R_p = R_p)
        res_array.append(dd)

#%% Read the Doppler error rate from the result of the previous cell
dprate = np.array([r["ErrorRate"]["Doppler"] 
                   for r in res_array]).reshape((len(parray), len(yarray)))

#%% Make a contour plot of the percentage of Doppler errors
plt.figure()
plt.contour(yarray*1e3, parray*1e3, dprate, np.arange(1,30,1), origin='lower')
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
filename = r"pitchRoll49PitchYawRelation.json"
with open(os.path.join(filepath, filename), "w") as f:
    f.write(json.dumps(res_array, indent=2))

#%% Run a simulation to calculate error rates for different points in the orbit
R_RPY = np.diag([4.8e-3, 0.4e-3, 1.1e-3])**2
res2_array = []
for k in range(len(mysvs)):
    X = mysvs[k][1]
    dd = eSim.simulateError(X, 
                            off_nadir, 
                            R_RPY = R_RPY, 
                            R_v = R_v, 
                            R_t = R_t, 
                            R_p = R_p)
    res2_array.append(dd)
    print("%0.2f" % (float(k+1)/len(mysvs)))

#%% Calculate the times at which the previous cell computed errors 
times = [(s[0] - mysvs[0][0])/np.timedelta64(1, 's') for s in mysvs]


#%% Save the generated simulation results to file
filename = r"sim-%s.json" % dt.now().strftime("%Y-%m-%dT%H%M%S")
with open(os.path.join(filepath, filename), "w") as f:
    f.write(json.dumps(res2_array, indent=2))

#%% Reload the file and plot the some results
with open(os.path.join(filepath, filename), "r") as f:
    res2 = json.loads(f.read())
    
dErrorRate = np.array([r["ErrorRate"]["Doppler"] for r in res2])
sErrorRate = np.array([r["ErrorRate"]["Swath"] for r in res2])
plt.figure()
plt.plot(times, dErrorRate)
plt.plot(times, sErrorRate)
plt.ylabel("Percentage in excess of threshold")
plt.xlabel("State Vector %s +Time (s)" % np.datetime_as_string(mysvs[0][0]))
plt.grid()
plt.show()
plt.legend(["Doppler", "Swath"])
print(np.mean(sErrorRate))
print(np.var(sErrorRate))
print(np.var(dErrorRate))
print(np.mean(dErrorRate))

