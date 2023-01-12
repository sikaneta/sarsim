# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:02:00 2022

@author: ishuwa.sikaneta
"""

from measurement.measurement import state_vector
import numpy as np
from scipy.constants import c
from space.planets import earth

#%% Define the range of elevation angles to look at
def getTiming(sv, elev, idx = 0):
    svdata = sv.measurementData[idx] 
    N = -svdata[:3]/np.linalg.norm(svdata[:3])
    T = svdata[3:]/np.linalg.norm(svdata[3:])
    C = np.cross(N, T)
    C = C/np.linalg.norm(C)
    T = np.cross(C,N)
    T = T/np.linalg.norm(T)
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
    XgSwath = np.insert(np.linalg.norm(Xg[1:,:] - Xg[:-1,:], axis=1).cumsum(), 0, 0,0)
    Xg = Xg*np.tile(1/np.linalg.norm(Xg, axis=1), (3,1)).T

    inc = np.degrees(np.arccos(-np.sum(Xg*rhat, axis=1)))
    
    return ranges, rhat, inc, tau, XgSwath

#%%
def surfaceNormal(XG, sv = state_vector(), planet = earth()):
    lat, lon, _ = sv.xyz2SphericalPolar(XG)
    clat = np.cos(np.radians(lat))
    slat = np.sin(np.radians(lat))
    clon = np.cos(np.radians(lon))
    slon = np.sin(np.radians(lon))
    
    n = np.array([planet.b*clat*clon,
                  planet.b*clat*slon,
                  planet.a*slat])
    
    return n/np.linalg.norm(n)

#%%
def satSurfaceGeometry(sv, off_nadir, azi_angle, planet = earth()):
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
        N = -surfaceNormal(s, sv, planet)
        C = np.cross(N,T)
        tcn_look = np.stack((T,C,N), axis=1).dot(look) 
        XG = sv.computeGroundPositionU(s, tcn_look)
        rvec[k,:] = s[0:3] - XG
        r[k] = np.linalg.norm(rvec[k,:])
        rhat = rvec[k,:]/r[k]
        snormal[k] = surfaceNormal(XG, sv, planet)
        incidence[k] = np.arccos(rhat.dot(snormal[k]))
        
    return rvec, snormal, r, incidence