# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:21:16 2022

@author: ishuwa.sikaneta
"""

from orbit.orientation import orbit
from measurement.measurement import state_vector
from orbit.geometry import satSurfaceGeometry
import numpy as np
from space.planets import earth
import matplotlib.pyplot as plt
from atmospheric.sensors import sensors
from scipy.constants import c

#%% Define parameters
sensorname = "wivern"
sensor = sensors[sensorname]

#%%
sorbit = sensor["orbit"]
period = 24*60*60/sorbit["orbits_per_day"]
t = np.arange(1001)/1001*period
planet = earth()
wivern_orb = orbit(e = sorbit["e"], 
                   inclination = sorbit["inclination"], 
                   arg_perigee = sorbit["arg_perigee"], 
                   period = period,
                   planet = earth(),
                   angleUnits="degrees")

#%%
base_time = np.datetime64("2022-12-06T18:00:00.000000000")
sv = state_vector()
llh = []
for tm in t:
    sv_time = base_time + np.timedelta64(int(tm*1e9), 'ns')
    sv_PCI = wivern_orb.computeSV(wivern_orb.computeO(tm)[0])[0]
    sv_PCR = sv.toPCR(sv_PCI, tm)
    llh.append(sv.xyz2polar(sv_PCR))
    sv.add(sv_time, sv_PCR) 

#%% plot some data
h = [l[2] for l in llh]
lat = [l[0] for l in llh]
speed = [np.linalg.norm(x[3:]) for x in sv.measurementData]
plt.figure()
plt.plot(t, h)
plt.grid()
plt.show()

#%%
plt.figure()
plt.plot(lat, h)
plt.grid()
plt.show()

#%%
plt.figure()
plt.plot(t, speed)
plt.grid()
plt.show()

#%%
plt.figure()
plt.plot(lat, speed)
plt.grid()
plt.show()

#%% Compute the look direction
off_nadir = np.radians(sensor["off_nadir"])
azi_angle = np.radians(90)
u = np.array([np.sin(off_nadir)*np.cos(azi_angle),
              np.sin(off_nadir)*np.sin(azi_angle),
              np.cos(off_nadir)])


#%% Compute the geometry
azi_angle = np.zeros(len(sv.measurementData),)
off_nadir = np.radians(sensor["off_nadir"])*np.ones(len(sv.measurementData),)
rvec, snoraml, r, incidence = satSurfaceGeometry(sv, off_nadir, azi_angle, planet)

#%% Compute the effective antenna length in elevation
B = 900e3
fc = 94.05e9
eta = 1.2
eLength = r*B/fc*eta*np.tan(incidence)


#%%
XG = sv.computeGroundPositionU(sv.measurementData[0], u)