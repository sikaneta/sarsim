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

#%% Load an oem orbit state vecctor file from Venus
orb_path = r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\Envision\Orbits"
svfile = os.path.join(orb_path, "EnVision_ALT_T4_2032_SouthVOI.oem")
with open(svfile, 'r') as f:
    svecs = f.read().split('\n')
svecs = [s.split() for s in svecs[16:-1]]
svecs = [s for s in svecs if len(s)==7]
svs = [(np.datetime64(s[0]), np.array([float(x) for x in s[1:]])*1e3) 
       for s in svecs]

#%% Run a test#%% Define a reference state vector to use
""" First get the state vector """
mysvs = eSim.state(svs, [270, 280, 10])
xidx = 0
X = mysvs[xidx][1]

#%%
# res2 = [simulateError(s[1], 30, sigmaA=0.022, sigmaE = 0.29, sigmaT = 0.12)
#         for s in mysvs]
off_nadir = -30

""" Generate the covariances """  
R_RPY = np.diag([4.9e-3, 0.2e-3, 1.4e-3])**2
R_v = np.diag([0.2, 0.2, 0.2])**2
R_t = 5**2
R_p = 430**2
    
#%%
R_RPY = np.diag([4.9e-3, 0.4e-3, 1.2e-3])**2
off_nadir = -30
eSim.simulateError(X, off_nadir, R_RPY = R_RPY, R_v = R_v, R_t = R_t, R_p = R_p)

#%%
parray = np.arange(0.1, 0.6, 0.05)*1.0e-3
yarray = np.arange(0.1, 2.0, 0.1)*1.0e-3

res_array = []
for p in parray:
    for y in yarray:
        print("Pitch: %0.2e, Yaw: %0.2e" % (p,y))
        R_RPY = np.diag([4.9e-3, p, y])**2
        dd = eSim.simulateError(X, off_nadir, R_RPY = R_RPY, R_v = R_v, R_t = R_t, R_p = R_p)
        res_array.append(dd)
    
# res2 = [simulateError(s[1], off_nadir, sigmaA=0.023, sigmaE = 0.26, sigmaT = 0.12) 
#         for s in mysvs]

#%%
dprate = np.array([r["ErrorRate"]["Doppler"] for r in res_array]).reshape((len(parray), len(yarray)))

#%%
plt.figure()
plt.contour(yarray*1e3, parray*1e3, dprate, np.arange(1,30,1), origin='lower')
plt.xlabel("yaw (mrad)")
plt.ylabel("pitch (mrad)")
plt.colorbar()
plt.grid()
plt.contour(yarray*1e3, parray*1e3, dprate, np.array([5]), origin='lower', colors = ['red'])
plt.title("Percentage of R-MIS-PER-1045 violations (Doppler)")
plt.show()

#%% Save and plot the error
filepath = r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\Envision\PointingSimulations"
filename = r"pitchRoll49PitchYawRelation.json"
with open(os.path.join(filepath, filename), "w") as f:
    f.write(json.dumps(res_array, indent=2))

#%% Compute angular velocity errors
Rv = np.diag([0.2, 0.2, 0.2])**2
for r,sv in zip(res2, mysvs):
    aerr = np.degrees(angularVelocityError(sv[1], off_nadir))
    r["computed"]["errors"]["sigmaAv"] = aerr[0]
    r["computed"]["errors"]["sigmaEv"] = aerr[1]
    r["computed"]["errors"]["sigmaTv"] = aerr[2]

#%% Compute angular timing errors
for r,sv in zip(res2, mysvs):
    aerr = np.degrees(angularTimingError(sv[1], off_nadir))
    r["computed"]["errors"]["sigmaAt"] = aerr[0]
    r["computed"]["errors"]["sigmaEt"] = aerr[1]
    r["computed"]["errors"]["sigmaTt"] = aerr[2]

#%%  
times = [(s[0] - mysvs[0][0])/np.timedelta64(1, 's') for s in mysvs]


#%%
filename = r"sim-%s.json" % dt.now().strftime("%Y-%m-%dT%H%M%S")
with open(os.path.join(filepath, filename), "w") as f:
    f.write(json.dumps(res2, indent=2))
    
#%% Set the filename
filename = r"sim-2022-06-17T150834.json"

#%%
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

#%%
sD = np.radians(np.array([r["computed"]["errors"]["sigmaDelta"] for r in res2]))*1e3
sAv = np.radians(np.array([r["computed"]["errors"]["sigmaAv"] for r in res2]))*1e3
sEv = np.radians(np.array([r["computed"]["errors"]["sigmaEv"] for r in res2]))*1e3
sTv = np.radians(np.array([r["computed"]["errors"]["sigmaTv"] for r in res2]))*1e3
sAt = np.radians(np.array([r["computed"]["errors"]["sigmaAt"] for r in res2]))*1e3
sEt = np.radians(np.array([r["computed"]["errors"]["sigmaEt"] for r in res2]))*1e3
sTt = np.radians(np.array([r["computed"]["errors"]["sigmaTt"] for r in res2]))*1e3

#%% Get the given values for each run
sE = np.radians(np.array([r["given"]["variances"]["sigmaE"] for r in res2]))*1e3
sA = np.radians(np.array([r["given"]["variances"]["sigmaA"] for r in res2]))*1e3
sT = np.radians(np.array([r["given"]["variances"]["sigmaT"] for r in res2]))*1e3

#%%
plt.figure()
plt.plot(times, sD)
plt.plot(times, sAv)
plt.plot(times, sEv)
plt.plot(times, sTv)
plt.plot(times, sAt, '--')
plt.plot(times, sEt, '--')
plt.plot(times, sTt, '--')
plt.ylabel("Error Standard Deviation (mrad)")
plt.xlabel("State Vector %s +Time (s)" % np.datetime_as_string(mysvs[0][0]))
plt.legend([r"$\sigma_{\epsilon_\delta}$", 
            r"$\sigma_{\alpha_v}$",
            r"$\sigma_{\epsilon_v}$",
            r"$\sigma_{\tau_v}$",
            r"$\sigma_{\alpha_t}$",
            r"$\sigma_{\epsilon_t}$",
            r"$\sigma_{\tau_t}$"])
plt.grid()
plt.show()
# %%
sElev = np.sqrt(sE**2 - (sD**2+sEv**2+sEt**2))
sAzimuth = np.sqrt(sA**2 - (sAv**2 + sAt**2))
sTilt = np.sqrt(sT**2 - (sTv**2 + sTt**2))

#%%
plt.figure()
plt.plot(times, sElev)
plt.plot(times, sAzimuth)
plt.plot(times, sTilt)
plt.ylabel("Margin (mrad)")
plt.xlabel("State Vector %s +Time (s)" % np.datetime_as_string(mysvs[0][0]))
plt.legend([r"$\sigma_{\epsilon_s}$", 
            r"$\sigma_{\alpha_s}$",
            r"$\sigma_{\tau_s}$"])
plt.grid()
plt.show()

# %% Stack plot for Elevation
fig, ax = plt.subplots()
errorE = {
    "position error": sD,
    "timing error": sEt,
    "velocity error": sEv,
    "margin": sElev
}
ax.stackplot(times, errorE.values(), labels=errorE.keys(), alpha=0.8)
ax.legend()
ax.set_title("Elevation Angle Error")
ax.set_xlabel("State Vector %s +Time (s)" % np.datetime_as_string(mysvs[0][0]))
ax.set_ylabel("Contributor (mrad)")
plt.show()
plt.show()

# %% Stack plot for Azimuth
fig, ax = plt.subplots()
errorE = {
    "timing error": sAt,
    "velocity error": sAv,
    "margin": sAzimuth
}
ax.stackplot(times, errorE.values(), labels=errorE.keys(), alpha=0.8)
ax.legend()
ax.set_title("Azimuth Angle Error")
ax.set_xlabel("State Vector %s +Time (s)" % np.datetime_as_string(mysvs[0][0]))
ax.set_ylabel("Contributor (mrad)")
plt.show()
plt.show()

# %% Stack plot for Tilt
fig, ax = plt.subplots()
errorE = {
    "timing error": sTt,
    "velocity error": sTv,
    "margin": sTilt
}
ax.stackplot(times, errorE.values(), labels=errorE.keys(), alpha=0.8)
ax.legend()
ax.set_title("Tilt Angle Error")
ax.set_xlabel("State Vector %s +Time (s)" % np.datetime_as_string(mysvs[0][0]))
ax.set_ylabel("Contributor (mrad)")
plt.show()
plt.show()

# %%
