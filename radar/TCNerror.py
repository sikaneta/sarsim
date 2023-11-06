# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:07:16 2023

@author: ishuwa.sikaneta
"""
import numpy as np
from orbit.orientation import orbit
from numpy.random import default_rng
rng = default_rng()
from orbit.elasticOrbit import getSVFromTime
from orbit.geometry import getTiming
import os
import json

#%% Define the json file with the covariances
def getOrbitCovariances(tag = "F1"):
    filepath = os.path.join(r"C:\Users",
                            r"ishuwa.sikaneta",
                            r"OneDrive - ESA",
                            r"Documents",
                            r"ESTEC",
                            r"Envision",
                            r"Orbits")
    with open(os.path.join(filepath, "EnvisionCovariance.json"), "r") as f:
        orbitCovariances = json.loads(f.read())
        
    try:
        return [x for x in orbitCovariances 
                if x["tag"] == tag][0]
    except IndexError:
        return None
    
#%% Function to return covariance matrix structure
def normalizeCovariance(R):
    std = np.sqrt(np.abs(np.diag(R)))
    crr = R/np.outer(std,std)
    return std, crr

#%%
def xtrackOffset2aeuCovarianceSwath(X,
                                    off_nadir,
                                    R_xtrack,
                                    sv):
    """
    Translate x-track position errors to elevation pointing error
    
    This function translates across-track satellite position errors into
    elevation angle pointing errors. The procedure follows the equation
    defined in section 6.5 of the PRJ

    Parameters
    ----------
    X : `np.ndarray(6, dtype=float)`
        State vector of the satellite at the time of interest.
    off_nadir : `float`
        The off-nadir angle for the elevation boresight.
    R_xtrack : `np.ndarray(2,2, dtype=float)`
        The covariance matrix of the across-track error in terms of
        errors in the c and n-directions, respecvely.

    Returns
    -------
    `float`
        Variance of the elevation angle error.

    """
    planetOrbit = orbit(planet=sv.planet, angleUnits="degrees")
    orbitAngle, ascendingNode = planetOrbit.setFromStateVector(X)
    tcn = planetOrbit.computeTCN(orbitAngle)
    
    r, rhat, inc, _, _ = getTiming(sv, [np.radians(off_nadir)], 0)
    
    d = np.cross(rhat[0], tcn[:,0]) - rhat[0]/np.tan(np.radians(inc[0]))
    
    q = d.dot(tcn[:,1:])#np.array([d.dot(c), d.dot(n)])
    
    return q.dot(R_xtrack).dot(q)/r[0]**2
    
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
def sigma(fld):
    return fld["value"]/fld["n_sigma"]

#%% 
def guidanceCovariance(tag = "F1"):
    myCV = getOrbitCovariances(tag)
    sv = getSVFromTime(myCV["UTC_time"], frame="VME2000")
    
    X = sv.measurementData[0]
    v = np.linalg.norm(X[3:])
    sigma_t = sigma(myCV["Tpos_error"])/v
    R, tcn = eCovariance((sigma_t)**2, X, sv)
    
    M_tcn = np.block([[tcn,np.zeros_like(tcn)],
                      [np.zeros_like(tcn),tcn]])
                      
    R_tcn = (M_tcn.T).dot(R).dot(M_tcn)
    
    std, crr = normalizeCovariance(R_tcn)
    Rv_tcn = np.diag([sigma(myCV["Tvel_error"])**2, 
                      sigma(myCV["Cvel_error"])**2, 
                      sigma(myCV["Nvel_error"])**2])
    new_cov = np.diag(Rv_tcn) - std[3:]**2
    new_cov[np.argwhere(new_cov<0)] = 0
    std_CN = np.array([sigma(myCV["Cpos_error"]),
                       sigma(myCV["Npos_error"])]) 
    return sigma_t**2, np.diag(std_CN**2 - std[1:3]**2), np.diag(new_cov)

#%%
# print(guidanceCovariance("F1"))