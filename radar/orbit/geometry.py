# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:02:00 2022

@author: ishuwa.sikaneta
"""

from measurement.measurement import state_vector
import numpy as np
from scipy.constants import c

#%% Define the range of elevation angles to look at
def getTiming(sv, elev, idx = 0):
    svdata = sv.measurementData[idx] 
    N = -svdata[:3]/np.linalg.norm(svdata[:3])
    T = svdata[3:]/np.linalg.norm(svdata[3:])
    C = np.cross(N, T)
    C = C/np.linalg.norm(C)
    N = np.cross(T, C)
    N = N/np.linalg.norm(N)
    # T = np.cross(C,N)
    # T = T/np.linalg.norm(T)
    uhats = np.array([np.cos(eang)*N + np.sin(eang)*C for eang in elev])

    # Calculate range vectors
    rangeVectors = sv.computeRangeVectorsU(svdata, uhats)

    # Calculate the ranges
    ranges = np.linalg.norm(rangeVectors, axis=1)
    rhat = rangeVectors*np.tile(1/ranges, (3,1)).T
    
    # Calculate the times
    tau = 2*ranges/c

    # Calculate the incidence angles
    Xg = np.tile(svdata[:3], (len(elev),1)) + rangeVectors
    snorm = np.array([surfaceNormal(x) for x in Xg])
    sgn = np.sign(np.sum(np.cross(T,snorm)*rhat, axis=1))
    XgSwath = -sgn*np.insert(np.linalg.norm(Xg[1:,:] - Xg[:-1,:], axis=1).cumsum(), 0, 0,0)
    #Xg = Xg*np.tile(1/np.linalg.norm(Xg, axis=1), (3,1)).T
    
    inc = np.degrees(sgn*np.arccos(-np.sum(snorm*rhat, axis=1)))
    
    return ranges, rhat, inc, tau, XgSwath

#%%
def surfaceNormal(XG, sv = state_vector()):
    lat, lon, _ = sv.xyz2SphericalPolar(XG)
    clat = np.cos(np.radians(lat))
    slat = np.sin(np.radians(lat))
    clon = np.cos(np.radians(lon))
    slon = np.sin(np.radians(lon))
    
    n = np.array([sv.planet.b*clat*clon,
                  sv.planet.b*clat*slon,
                  sv.planet.a*slat])
    
    return n/np.linalg.norm(n)

#%%
def satSurfaceGeometry(sv, off_nadir, azi_angle):
    soff_nadir = np.sin(off_nadir)
    coff_nadir = np.cos(off_nadir)
    sazi_angle = np.sin(azi_angle)
    cazi_angle = np.cos(azi_angle)
    
    u = np.stack((soff_nadir*cazi_angle,
                  soff_nadir*sazi_angle,
                  coff_nadir), axis=1)
    
    r = np.zeros_like(azi_angle)
    incidence = np.zeros_like(azi_angle)
    rvec = np.zeros((len(azi_angle), 3), dtype=float)
    snormal = np.zeros_like(rvec)
    
    for k in range(len(azi_angle)):
        s = sv.measurementData[k]
        look = u[k]
        T = s[3:]/np.linalg.norm(s[3:])
        N = -surfaceNormal(s, sv)
        C = np.cross(N,T)
        tcn_look = np.stack((T,C,N), axis=1).dot(look) 
        XG = sv.computeGroundPositionU(s, tcn_look)
        rvec[k,:] = s[0:3] - XG
        r[k] = np.linalg.norm(rvec[k,:])
        rhat = rvec[k,:]/r[k]
        snormal[k] = surfaceNormal(XG, sv)
        incidence[k] = np.arccos(rhat.dot(snormal[k]))
        
    return rvec, snormal, r, incidence


#%% New find nearest based on ordered time
def findNearest(timeArray, eta):
    nElements = len(timeArray)
    def new_bracket(bracket):
        center = int((bracket[1] + bracket[0])/2)
        if timeArray[center] < eta:
            return center, bracket[1]
        else:
            return bracket[0], center
    
    bracket = new_bracket([0, nElements])
    while bracket[1] - bracket[0] > 1:
        bracket = new_bracket(bracket)
        
    if (np.abs(eta - timeArray[bracket[0]]) < 
        np.abs(eta - timeArray[bracket[1]])):
        return bracket[0]
    else:
        return bracket[1]

#%%
def computeImagingGeometry(sv, eta, xG, xG_snormal):
    idx = findNearest(sv.measurementTime, eta)
    
    mysv = state_vector(planet=sv.planet)
    mysv.add(sv.measurementTime[idx], sv.measurementData[idx])
    
    svtime, svdata, err = mysv.computeBroadsideToX(eta, xG)
    
    N = -svdata[:3]/np.linalg.norm(svdata[:3])
    T = svdata[3:]/np.linalg.norm(svdata[3:])
    C = np.cross(N, T)
    C = C/np.linalg.norm(C)
    N = np.cross(T, C)
    N = N/np.linalg.norm(N)
    
    
    rvec = xG - svdata[0:3] 
    rhat = rvec/np.linalg.norm(rvec)
    sgn = np.sign(np.cross(T,xG_snormal).dot(rhat))
    
    incidence = np.degrees(sgn*np.arccos(-rhat.dot(xG_snormal)))
    
    return rvec, incidence, [svtime, svdata, err]
