#%% -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:10:22 2022

@author: Ishuwa.Sikaneta
"""

from orbit.envision import envisionState, angularVelocityError
from orbit.envision import simulateError, angularTimingError
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
import json
from datetime import datetime as dt

rng = default_rng()
import os


#%% Define some parameters for venSAR

elAxis = 0.6
azAxis = 6.0

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
mysvs = envisionState(svs, [270, 480, 10])
xidx = 0
X = mysvs[xidx][1]

#%%
# res2 = [simulateError(s[1], 30, sigmaA=0.022, sigmaE = 0.29, sigmaT = 0.12)
#         for s in mysvs]
off_nadir = 30
res2 = [simulateError(s[1], off_nadir, sigmaA=0.023, sigmaE = 0.26, sigmaT = 0.12) 
        for s in mysvs]

#%% Compute angular velocity errors
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

#%% Save and plot the error
filepath = r"c:\Users\Ishuwa.Sikaneta\Documents\ESTEC\Envision"

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
