# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:05:34 2023

@author: ishuwa.sikaneta
"""

import json
from glob import glob
import os
import numpy as np
from orbit.geometry import findNearest
import tqdm

#%% Define the folder containing data
filepath = r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\Envision\ROIs"

#%% Open the Jayne file
with open(os.path.join(filepath, "quindec_cycle_orbit_table.txt"), "r") as f:
    pool = [x.split() for x in f.read().split("\n")]
    pool = pool[0:-1]
    
#%% Load all orbits
pp = [[np.datetime64(x[0]), int(x[1]), int(x[2]), int(x[3])] for x in pool]

#%% Filter for orbits where SAR is possible
ppsarstd = [p for p in pp if p[2] in [1,6,11]]
sarorbits = [p[1] for p in ppsarstd]

#%% Test values
tt = [np.datetime64("2038-06-08T04:38:00.509329858")]
tt += [np.datetime64("2035-07-18T20:33:06.662090361")]
tt += [np.datetime64("2035-03-17T03:52:45.9999")]
print([[(k,x) for k,x in enumerate(pp) if x[0]>t][0] for t in tt])

#%% Load up data and correct orbit numbers
incpath = r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\Envision\ROIs\incidence"
flist = glob(os.path.join(incpath, "*_incidence.geojson"))
for fl in tqdm(flist):
    with open(fl, "r") as f:
        pool = json.loads(f.read())
    for feat in pool["features"]:
        utc = np.datetime64(feat["properties"]["stateVector"]["J2000"]["time"])
        feat["properties"]["orbitNumber"] = eI.getOrbitNumber([utc])[0]
        feat["properties"]["SAROrbitFilter"] = int(feat["properties"]["orbitNumber"] in sarorbits)
    with open(fl, "w") as f:
        f.write(json.dumps(pool, indent=2))
        
#%% Function to return Toeplitz matrix
def myTop(seed, N):
    M = len(seed)
    fseed = seed[-1:0:-1] + seed
    Mf = len(fseed)
    R = np.zeros((N,N+Mf-1))
    for k in range(N):
        R[k,k:(k + Mf)] = fseed
    R = R[:,(M-1):(-M+1)]

    D, U = np.linalg.eig(R)
    plt.figure()
    plt.plot(D,'.')
    plt.grid()
    plt.show()
    
    plt.figure()
    plt.plot(U[:,0:min(4,N)])
    plt.grid()
    plt.show()
    
    return R, D, U