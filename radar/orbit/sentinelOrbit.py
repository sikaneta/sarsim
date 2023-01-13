# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 09:45:38 2023

@author: ishuwa.sikaneta
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:21:16 2022

@author: ishuwa.sikaneta
"""

from orbit.orientation import orbit
from orbit.geometry import getTiming
from measurement.measurement import state_vector
import numpy as np
import matplotlib.pyplot as plt

#%%
sentinel_orb = orbit(e = 0.0001303, 
                     arg_perigee = np.radians(79.7417), 
                     a = 7167100, 
                     inclination = np.radians(98.1813))

#%%
t = np.arange(1001)/1001*sentinel_orb.period

#%%
base_time = np.datetime64("2022-12-06T18:00:00.000000000")
sv = state_vector()
llh = []
for tm in t:
    sv_time = base_time + np.timedelta64(int(tm*1e9), 'ns')
    sv_PCI = sentinel_orb.computeSV(sentinel_orb.computeO(tm)[0])[0]
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
""" Compute Minimum altitude """
min_idx = np.argmin(h)
print(llh[np.argmin(h)])

#%% Generate some data
off_nadir = np.arange(0.01,60,0.01)

#%%
ranges, _, incidence, _, g_ranges = getTiming(sv, np.radians(off_nadir), idx = min_idx)

#%% Plot the data
plt.figure()
plt.plot(incidence, ranges, incidence, g_ranges)
plt.xlabel("Incidence angle (deg)")
plt.ylabel("Range and Ground_range (m)")
plt.legend(["Range", "Ground Range"])
plt.grid()
plt.show()

#%%
idx_20 = np.argmin(np.abs(incidence - 20))
idx_45 = np.argmin(np.abs(incidence - 45))
print("Ground swath: %0.2f" % ((g_ranges[idx_45] - g_ranges[idx_20])/1e3))
