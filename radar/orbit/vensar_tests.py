#%% -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:10:22 2022

@author: Ishuwa.Sikaneta
"""

from space.planets import venus
from orbit.orientation import orbit
from orbit.euler import YRPfromRotation
from orbit.euler import aeuAnglesDAEUfromAAEU, aeuAnglesAAEUfromDAEU
import numpy as np
import matplotlib.pyplot as plt
from measurement.measurement import state_vector
from scipy.constants import c
from numpy.random import default_rng
import json
from datetime import datetime as dt

rng = default_rng()
import os

#%% Define a function to estimate the PDF given a histogram
def estimatePDF(d, N=400):
    h,x = np.histogram(d,N)
    x = (x[:-1] + x[1:])/2
    h = h/(np.mean(np.diff(x))*len(d))
    return h,x

#%% Define some parameters for venSAR

elAxis = 0.6
azAxis = 6.0

#%% Load an oem orbit state vecctor file from Venus
orb_path = r"C:\Users\Ishuwa.Sikaneta\local\Data\Envision"
svfile = os.path.join(orb_path, "EnVision_ALT_T4_2032_SouthVOI.oem")
with open(svfile, 'r') as f:
    svecs = f.read().split('\n')
svecs = [s.split() for s in svecs[16:-1]]
svecs = [s for s in svecs if len(s)==7]
svs = [(np.datetime64(s[0]), np.array([float(x) for x in s[1:]])*1e3) 
       for s in svecs]

#%% Define a function that will return state vectors in VCI coordinates
""" State vectors provided are in a coordinate system defined
    by AOCS folks. The VCI coordinate system, as per Envision 
    definitions, defines the j vector in the direction of the
    ascending node. Thus, we need to find the components of the
    state vectors in this coordinate system. """
def envisionState(svs, idx):
    venSAR = orbit(planet=venus(), angleUnits="radians")
    omega = np.array([venSAR.state2kepler(svs[k][1])["ascendingNode"] 
                      for k in range(idx[0],idx[1])]).mean()
    co = np.cos(omega)
    so = np.sin(omega)
    Mo = np.array([[co, so, 0],[-so,co,0],[0,0,1]])
    Moo = np.block([[Mo, np.zeros_like(Mo)], [np.zeros_like(Mo), Mo]])
    
    return [(svs[k][0], svs[k][1].dot(Moo)) for k in range(idx[0],idx[1])]

#%% angular error from velocity error
def angularVelocityError(X, 
                         off_nadir,
                         sigmaVX = 0.2, 
                         sigmaVY = 0.2, 
                         sigmaVZ = 0.2, 
                         N = 10000):
    
    """ Define the off-nadir directional cosine """
    v = np.cos(np.radians(off_nadir))
    
    """ Create an orbit around Venus """
    venSAR = orbit(planet=venus(), angleUnits="degrees")
    
    """ Generate random velocities """
    SVe = np.vstack((np.zeros((3,N)),
                     np.diag([sigmaVX, 
                              sigmaVY, 
                              sigmaVZ]).dot(rng.standard_normal((3,N)))))
    
    """ Compute the reference aueIe frame that is in error 
        from wrong velocity """
    orbitAngle, ascendingNode = venSAR.setFromStateVector(X)
    aaeu, _ = venSAR.computeAEU(orbitAngle, v)
    
    """ Compute the desired aeuI frame for each true velocity """
    daeu = np.zeros((N,3,3), dtype = np.double)
    for k in range(N):
        orbitAngle, ascendingNode = venSAR.setFromStateVector(X + SVe[:,k])
        daeu[k,:,:], _ = venSAR.computeAEU(orbitAngle, v)
        
    """ Compute the tilt, elevation and azimuth angles """
    AEU = aeuAnglesDAEUfromAAEU(daeu, aaeu)
    
    """ Return computed standard deviations """
    return np.sqrt(np.mean(AEU**2, axis=0))
    

#%% timing error
def angularTimingError(X, off_nadir, dT=5, N=10000):
    
    """ Define the off-nadir directional cosine """
    v = np.cos(np.radians(off_nadir))
    
    """ Create an orbit around Venus """
    venSAR = orbit(planet=venus(), angleUnits="degrees")
    
    """ Compute the orbit angle for the given state vector"""
    orbitAngle, ascendingNode = venSAR.setFromStateVector(X)
    
    """ Compute constant relating derivative of orbit angle to derivative 
        of time """
    e = venSAR.e
    GM = venSAR.planet.GM
    a = venSAR.a
    U = np.radians(orbitAngle) - venSAR.arg_perigee
    C = (1+e*np.cos(U))**2*np.sqrt(GM/
                         (a*(1-e**2))**3)
    
    """ Generate random orbit angle deviances """
    dO = rng.standard_normal(N)*dT*C
    
    """ Calculate the DAEU frame """
    daeu, _ = venSAR.computeAEU(orbitAngle, v)
    
    """ Loop over deviations in orbit angle """
    aaeu = np.zeros((N,3,3), dtype = np.double)
    for k in range(N):
        aaeu[k,:,:], _ = venSAR.computeAEU(orbitAngle + dO[k], v)
        
    """ Compute the tilt, elevation and azimuth angles """
    AEU = aeuAnglesAAEUfromDAEU(aaeu, daeu)
    
    """ Return computed standard deviations """
    return np.sqrt(np.mean(AEU**2, axis=0))
    
    
#%%
def simulateError(X, 
                  off_nadir,
                  sigmaA = 0.023,
                  sigmaE = 0.26,
                  sigmaT = 0.12,
                  sigmaP = 600,
                  e_ang = 14.28,
                  azAxis = 6.0,
                  elAxis = 0.6,
                  carrier = 3.15e9,
                  nA = 300,
                  nE = 300,
                  nT = 100,
                  loglevel = 0):
    
    """ Define a return dictionary """
    res = {"given": {"off_nadir": off_nadir,
                     "azAxis": azAxis,
                     "elAxis": elAxis,
                     "carrier": carrier,
                     "variances": {
                         "sigmaA": sigmaA,
                         "sigmaE": sigmaE,
                         "sigmaT": sigmaT,
                         "sigmaP": sigmaP
                         }
                     }
           }
    
    wavelength = c/carrier
    elBW = wavelength/elAxis
    azBW = wavelength/azAxis
    
    res["computed"] = {"wavelength": wavelength,
                       "beamwidths": {"units": "degrees",
                                      "azimuth": np.degrees(azBW),
                                      "elevation": np.degrees(elBW)}
                       }
    
    """ Create an orbit around Venus """
    venSAR = orbit(planet=venus(), angleUnits="degrees")
    orbitAngle, ascendingNode = venSAR.setFromStateVector(X)
    
    """ Define the off-nadir angle """
    v = np.cos(np.radians(off_nadir))
    
    XI, rI, vI = venSAR.computeR(orbitAngle)
    res["State Vector Radius"] = np.linalg.norm(X[:3])
    res["Kepler"] = {"radius": np.linalg.norm(rI),
                     "a": venSAR.a,
                     "e": venSAR.e,
                     "period": venSAR.period,
                     "altitude": np.linalg.norm(rI) - venSAR.planet.a,
                     "angles": {"units": "degrees",
                                "orbit": orbitAngle,
                                "ascending node": ascendingNode,
                                "inclination": np.degrees(venSAR.inclination),
                                "perigee": np.degrees(venSAR.arg_perigee)}
                     }
    e1, e2 = venSAR.computeE(orbitAngle, v)

    aeuI, aeuTCN = venSAR.computeAEU(orbitAngle, v)
    
    """ Compute the rotation matrix to go from aeu to ijk_s """
    ep_ang = np.radians(e_ang)
    c_ep = np.cos(ep_ang)
    s_ep = np.sin(ep_ang)
    M_e = np.array([
                    [0,     1,     0],
                    [s_ep,  0,  c_ep],
                    [c_ep,  0, -s_ep]
                   ])
    ijk_s = aeuI.dot(M_e) 
    (theta_y, theta_r, theta_p), _ = venSAR.YRPfromRotation(ijk_s)

    M, dM = venSAR.computeItoR(orbitAngle)
    XP = np.hstack((M.dot(XI[0:3]), dM.dot(XI[0:3]) + M.dot(XI[3:])))
    vP = np.linalg.norm(XP[3:])
    dopBW = 2*vP/wavelength*azBW
    aeuP = M.dot(aeuI)
    vAngles = np.arange(-np.degrees(elBW/2), np.degrees(elBW/2), 0.01)

    if loglevel > 2:
        print("Norm of u: %0.6f" % np.linalg.norm(aeuTCN[:,2]))
        print("u*e1: %0.6f" % np.dot(aeuTCN[:,2],e1))
        print("u*e2: %0.6f, v: %0.6f" % (np.dot(aeuTCN[:,2],e2), v))
        print("uP*VP: %0.6f" % XP[3:].dot(aeuP[:,2]))
        print("Pitch: %0.4f, Roll: %0.4f, Yaw: %0.4f" % (np.degrees(theta_p),
                                                         np.degrees(theta_r),
                                                         np.degrees(theta_y)))
        
    res["attitude"] = {"ikj_s": ijk_s.tolist(),
                       "aeuI": aeuI.tolist(),
                       "aeuP": aeuP.tolist(),
                       "PRY": {"units": "degrees",
                               "pitch": np.degrees(theta_p),
                               "roll": np.degrees(theta_r),
                               "yaw": np.degrees(theta_y)}
                       }

    """ Compute the time """
    t = venSAR.computeT(orbitAngle)
    res["Kepler"]["time"] = t
        
    """ Instantiate a state vector object """
    sv = state_vector(planet=venus(), harmonicsCoeff=180)
    
    """ Compute ECEF state vector """ 
    XP = sv.toPCR(XI, t)
    
    """ Compute the ground position """
    XG = sv.computeGroundPositionU(XP, aeuP[:,2])
    
    """ Test the ground position """
    R = XG - XP[:3]
    rhat = R/np.linalg.norm(R)
    vG = -XP[:3].dot(rhat)/np.linalg.norm(XP[:3])
    pG = (XG[0]/venSAR.planet.a)**2 + (XG[1]/venSAR.planet.a)**2 + (XG[2]/venSAR.planet.b)**2
    dG = XP[3:].dot(rhat)
    res["VCR"] = {"X": list(XG),
                  "cosine off-nadir": vG,
                  "Doppler Centroid": dG,
                  "EllipsoidFn": pG}
    
    """ Generate random errors """    
    alpha_list = sigmaA*rng.standard_normal(nA)
    epsilon_list = sigmaE*rng.standard_normal(nE)
    tau_list = sigmaT*rng.standard_normal(nT)
    
    """ Generate the AEU vectors with the errors """
    aeuPe = venSAR.pointingErrors(aeuP, alpha_list, epsilon_list, tau_list)
    
    """ Compute the Doppler centroids for the min and max elevation angles """
    dc = venSAR.dopCens(aeuPe, [vAngles[0], vAngles[-1]], XP[3:], wavelength)
    
    """
    This mapping of a function along the last axis allows computation of the
    greatest difference of the Doppler centroid from zero. It isn't particularly
    fast, but also not too slow."""
    dc_max = np.max(np.abs(dc), axis=-1)

    """ Plot the histogram of the maximum Doppler centroids across the beam. """
    N = np.prod(list(dc_max.shape))
    hD, xD = estimatePDF(dc_max.reshape((N,)), N=200)
    idxD = np.argwhere(xD > 1/15*dopBW)
    idxD = idxD[0][0] if idxD.size > 0 else len(xD)
    dError = 100*np.sum(hD[idxD:])*(xD[1]-xD[0])
    if loglevel > 2:
        print("Percent of Doppler centroids in violation of zero-Doppler threshold: %0.4f" % dError)
        
    if loglevel > 1:
        plt.figure()
        plt.plot(xD, hD)
        plt.axvline(x = 1/15*dopBW, color = 'r', label = '1/15 of Doppler Bandwdith')
        plt.xlabel('Doppler centroid (Hz)')
        plt.ylabel('Histogram count')
        mytitle = r"Max $f_{dc}$, $\sigma_{\alpha}=%0.3f$, $\sigma_{\epsilon}=%0.3f$, $\sigma_{\tau}=%0.3f$ (deg)" % (sigmaA, sigmaE, sigmaT)
        plt.title(mytitle)
        plt.grid()
        plt.show()


    """ Compute beam projection differences """
    off_boresight = [vAngles[0], 0, vAngles[-1]]
    vS = np.array([[0, 
                    np.sin(np.radians(ob)),
                    np.cos(np.radians(ob))] for ob in off_boresight])
    
    """ Compute the reference min and max ranges """
    refR0 = sv.computeRangeVectorsU(XP, np.matmul(aeuP, vS[0,:]))
    refR1 = sv.computeRangeVectorsU(XP, np.matmul(aeuP, vS[1,:]))
    refR2 = sv.computeRangeVectorsU(XP, np.matmul(aeuP, vS[2,:]))
    refRng0 = np.linalg.norm(refR0, axis=-1)
    refRng1 = np.linalg.norm(refR1, axis=-1)
    refRng2 = np.linalg.norm(refR2, axis=-1)
    
    """ Compute the perturbed min and max ranges """
    R0 = sv.computeRangeVectorsU(XP, np.matmul(aeuPe, vS[0,:]))
    R1 = sv.computeRangeVectorsU(XP, np.matmul(aeuPe, vS[1,:]))
    R2 = sv.computeRangeVectorsU(XP, np.matmul(aeuPe, vS[2,:]))
    
    Rng0 = np.linalg.norm(R0, axis=-1)
    Rng1 = np.linalg.norm(R1, axis=-1)
    Rng2 = np.linalg.norm(R2, axis=-1)
    
    """ define vectorized functions to compute min and max over array """
    myfmin = np.vectorize(lambda x, y: x if x < y else y)
    myfmax = np.vectorize(lambda x, y: x if x > y else y)
    coverage = (myfmin(Rng2, refRng2) - myfmax(Rng0, refRng0))/(refRng2 - refRng0)
    
    hS, xS = estimatePDF(coverage.reshape((N,)), N=50)
    idxS = np.argwhere(xS < 14/15)[-1][0]
    sError = 100*np.sum(hS[0:idxS])*(xS[1]-xS[0])
    if loglevel > 2:
        print("Percent of Swaths in violation of overlap threshold: %0.4f" % sError)
        
    if loglevel > 1:
        plt.figure()
        plt.plot(xS, hS)
        plt.axvline(x = 14/15, color = 'r', label = '14/15 of beamwidth')
        plt.xlabel('Fraction of elevation beam overlap on ground (unitless).')
        plt.ylabel('Histogram count')
        mytitle = r"Swath overlap ratio, $\sigma_{\alpha}=%0.3f$, $\sigma_{\epsilon}=%0.3f$, $\sigma_{\tau}=%0.3f$ (deg)" % (sigmaA, sigmaE, sigmaT)
        plt.title(mytitle)
        plt.grid()
        plt.show()
    res["ErrorRate"] = {"Doppler": dError,
                        "Swath": sError}
    
    del aeuPe

    """ Calculate error in epsilon due to orbit errors """
    sigmaEs = np.degrees(sigmaP/np.linalg.norm(R))
    
    alpha_list = sigmaA*rng.standard_normal(nA)
    epsilon_list = (sigmaE - sigmaEs)*rng.standard_normal(nE)
    tau_list = sigmaT*rng.standard_normal(nT)
    
    """ Generate the ijk vectors with the errors in the SC frame """
    ijk_se = venSAR.pointingErrors(aeuI, alpha_list, epsilon_list, tau_list).dot(M_e)
        
    """ Flatten arrays so that we can use numba funtion """
    YRP = np.zeros((N, 3), dtype=np.double)
    YRPfromRotation(ijk_se.reshape((N,3,3)), ijk_s, YRP)
    
    """ Calculate the reference roll pitch and roll """
    (pitch_0, roll_0, yaw_0), _ = venSAR.YRPfromRotation(ijk_s)
    
    """ Calculate the variances in YRP """
    vAET = np.array([sigmaA, sigmaE - sigmaEs, sigmaT])/180*np.pi*1e3
    vYRP = np.sqrt(np.mean(YRP**2, axis=0))*1e3
    if loglevel > 2:
        print("Standard deviation in Alpha, Epsilon, Tau:")
        print(vAET)
        print("Standard deviation in Yaw, Roll, Pitch:")
        print(vYRP)
        
    res["pointingErrors"] = {"units": "mrad",
                             "AlphaEpsilonTau": list(vAET),
                             "YawRollPitch": list(vYRP)}
    
    res["computed"]["errors"] = {"sigmaDelta": sigmaEs}
    
    if loglevel > 0:
        print(json.dumps(res, indent=2))
    
    return res

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
