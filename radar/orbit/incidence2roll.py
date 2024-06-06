# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:00:08 2024

@author: ishuwa.sikaneta
"""

from space.planets import venus
from orbit.pointing import simulation
from orbit.geometry import getTiming
from orbit.envision import loadSV
from orbit.elasticOrbit import getSVFromTime, client
from TCNerror import guidanceCovariance, getOrbitCovariances
from measurement.measurement import state_vector
from itertools import product
from space.planets import venus
from orbit.orientation import orbit

from orbit.euler import rpyAnglesFromIJK


""" System Packages """
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime as dt
import os
from scipy.optimize import root

#%%
mode = {
    "StandardSAR": {
        "axes": np.eye(3),
        "off_nadirs": [18.7, 27.8, 36.9, -18.7, -27.8, -36.9]
    },
    "Altimeter": {
        "axes": np.array([[0,1,0],[1,0,0],[0,0,1]]),
        "off_nadirs": [0.001]
    },
    "Near-Nadir Radiometry": {
        "axes": np.eye(3),
        "off_nadirs": [-14.28]
    },
    "Nadir Radiometry": {
        "axes": np.array([[0,1,0],[1,0,0],[0,0,1]]),
        "off_nadirs": [0.001]
    }
}
modestr = "StandardSAR"
eSim = simulation(planet = venus(),
                  e_ang = 14.28, # The effective angle of the beam relative to spacecraft coordinate frame
                  azAxis = 5.5,  # The length of the reflector in the azimuth direction. Leads to 0.99 degrees
                  elAxis = 0.6125,  # The length of the reflector in the elevation direction. Leads to 8.9 degrees
                  carrier = 3.15e9,
                  mode = mode[modestr]["axes"])

#%%
tag = "F1"
svTime = "2039-01-10T00:00:00"
sv = getSVFromTime(svTime, frame="VME2000")
#X = sv.measurementData[0]

#%%
def offNadir2incidence(x, sv):
    _, _, inc, _, _ = getTiming(sv, [np.radians(x)])
    return inc[0]

#%%
def esimateSVFromPerigee(svTime, 
                         arg_from_perigee = 0, 
                         time_error=1e-2,
                         max_iter = 10):
    
    """ Create an orbit around Venus """
    sv = getSVFromTime(svTime, frame="VME2000")
    planetOrbit = orbit(planet=sv.planet, angleUnits="degrees")
    
    satX = sv.measurementData[0]
    orbitAngle, ascendingNode = planetOrbit.setFromStateVector(satX)
    
    dTzero = planetOrbit.computeT(np.degrees(planetOrbit.arg_perigee) + 
                                  arg_from_perigee)
    
    nTime = sv.measurementTime[0]
    idx = 0
    dT = 1.0
    orbitPeriods = planetOrbit.period*np.arange(-1,2)
    
    while abs(dT) > time_error and idx < max_iter:
        idx += 1
        orbitPeriods = planetOrbit.period*np.arange(-1,2)
        dT = planetOrbit.computeT(orbitAngle) - dTzero
        dTshift = np.argmin(abs(dT + orbitPeriods))
        dT += orbitPeriods[dTshift]
        nTime -= np.timedelta64(int(dT*1e3), 'ms')
        
        satX = sv.estimate(nTime)
        orbitAngle, ascendingNode = planetOrbit.setFromStateVector(satX)
    
    return satX, nTime, planetOrbit
    
#%%
def offNadir2steer(off_nadir, satX, planetOrbit, sv, eSim):
    # """ Create an orbit around Venus """
    # planetOrbit = orbit(planet=venus(), angleUnits="degrees")
    
    # satX = sv.measurementData[0]
    orbitAngle, ascendingNode = planetOrbit.setFromStateVector(satX)
    
    """ Define the off-nadir angle """
    v = np.cos(np.radians(off_nadir))
    
    XI, rI, vI = planetOrbit.computeSV(orbitAngle)
    e1, e2 = planetOrbit.computeE(orbitAngle, v)
    
    aeuI, aeuTCN = planetOrbit.computeAEU(orbitAngle, off_nadir)
    tcnI = aeuI.dot(aeuTCN.T)
    
    """ Compute the rotation matrix to go from aeu to ijk_s """
    c_ep = np.cos(eSim.e_ang)
    s_ep = np.sin(eSim.e_ang)
    M_e = np.array([
                    [0,     1,     0],
                    [s_ep,  0,  c_ep],
                    [c_ep,  0, -s_ep]
                   ])
    ijk_s = aeuI.dot(M_e) 
    qq = aeuTCN.dot(M_e)
    
    """ Compute the Euler angles """
    (theta_r, 
     theta_p, 
     theta_y) = rpyAnglesFromIJK(qq.reshape((1,3,3))).flatten()
    
    M, dM = planetOrbit.computeItoR(orbitAngle)
    # XP = np.hstack((M.dot(XI[0:3]), dM.dot(XI[0:3]) + M.dot(XI[3:])))
    # vP = np.linalg.norm(XP[3:])
    # dopBW = 2*vP/eSim.wavelength*eSim.azBW
    # aeuP = M.dot(aeuI)
    
    # print("Norm of u: %0.6f" % np.linalg.norm(aeuTCN[:,2]))
    # print("u*e1: %0.6f" % np.dot(aeuTCN[:,2],e1))
    # print("u*e2: %0.6f, v: %0.6f" % (np.dot(aeuTCN[:,2],e2), v))
    # print("uP*VP: %0.6f" % XP[3:].dot(aeuP[:,2]))
        
    cval = {
           # "ijk_s": ijk_s.tolist(),
           # "tcnI": tcnI.tolist(),
           # "aeuI": aeuI.tolist(),
           # "aeuP": aeuP.tolist(),
           "altitude": sv.xyz2polar(satX)[-1],
           "RPY": {
               "units": "degrees",
               "roll": np.degrees(theta_r),
               "pitch": np.degrees(theta_p),
               "yaw": np.degrees(theta_y)
               }
           }
    return cval

#%%
steer = [  
    {
     "tag": "220km altitude",
     "angleFromPerigee": 0.0,
     "UTCtime": "2039-01-17T09:17:23",
     "cases": [
         {"incidence": 22.5},
         {"incidence": 24.5},
         {"incidence": 35.5},
         {"incidence": 37.5}
     ]
    },
    {
     "tag": "510km altitude",
     "angleFromPerigee": 180.0,
     "UTCtime": "2039-01-17T17:51:23",
     "cases": [
         {"incidence": 22.5},
         {"incidence": 24.5},
         {"incidence": 35.5},
         {"incidence": 37.5}
     ]
    }
]

for point in steer:
    sv = getSVFromTime(point["UTCtime"], frame="VME2000")
    satX, nTime, planetOrbit = esimateSVFromPerigee(svTime, 
                                                    arg_from_perigee = point["angleFromPerigee"])
    
    for case in point["cases"]:
        incidence = case["incidence"]
        fz = root(lambda x: offNadir2incidence(x,sv) - incidence, incidence)
        off_nadir = fz.x[0]
        left = offNadir2steer(off_nadir, satX, planetOrbit, sv, eSim)
        left["offNadir"] = off_nadir
        right = offNadir2steer(-off_nadir, satX, planetOrbit, sv, eSim)
        right["offNadir"] = -off_nadir
        case["left"] = left
        case["right"] = right
        
#%%
print(json.dumps(steer, indent=2))
with open("spaceCraftRollValues.json", "w") as f:
    f.write(json.dumps(steer, indent=2))
