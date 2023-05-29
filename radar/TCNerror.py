# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:07:16 2023

@author: ishuwa.sikaneta
"""
import numpy as np
from orbit.orientation import orbit
from numpy.random import default_rng
rng = default_rng()
from orbit.envision import loadSV


#%% Function to return covariance matrix structure
def normalizeCovariance(R):
    std = np.sqrt(np.abs(np.diag(R)))
    crr = R/np.outer(std,std)
    return std, crr

#%%
def eCovariance(Rt, X, sv, N=100000):
    planetOrbit = orbit(planet=sv.planet, angleUnits="degrees")
    orbitAngle, ascendingNode = planetOrbit.setFromStateVector(X)
    tcn = planetOrbit.computeTCN(orbitAngle)
    e = planetOrbit.e
    GM = planetOrbit.planet.GM
    a = planetOrbit.a
    U = np.radians(orbitAngle) - planetOrbit.arg_perigee
    C = (1+e*np.cos(U))**2*np.sqrt(GM/
                         (a*(1-e**2))**3)

    dO = rng.standard_normal(N)*np.sqrt(Rt)*np.degrees(C)
    dX = np.array([planetOrbit.computeSV(orbitAngle + o)[0] - X for o in dO])
    return dX.T.dot(dX)/N, tcn



#%%
sv = loadSV()

#%%
X = sv.measurementData[145801]
R, tcn = eCovariance((0.5716)**2, X, sv)

M_tcn = np.block([[tcn,np.zeros_like(tcn)],
                  [np.zeros_like(tcn),tcn]])
                  
R_tcn = (M_tcn.T).dot(R).dot(M_tcn)

std, crr = normalizeCovariance(R_tcn)
Rv_tcn = np.array([[0.337,0,0],
                   [0,0.951,0],
                   [0,0,13.840]])
R_res = np.diag(np.sqrt(np.diag(Rv_tcn)**2 
        - (3*std[3:])**2))
print(R_res)