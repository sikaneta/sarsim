# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:30:18 2022

@author: ishuwa.sikaneta
"""
import numpy as np
from space.planets import venus
from orbit.orientation import orbit
from orbit.euler import YRPfromRotation, RPYfromRotation
from orbit.euler import rpyAnglesFromIJK
from orbit.euler import aeuAngles
from orbit.euler import RPYfromAEU, AEUfromRPY
from orbit.euler import aeu2rot
#from orbit.euler import aeuAnglesDAEUfromAAEU, aeuAnglesAAEUfromDAEU
from measurement.measurement import state_vector
from numpy.random import default_rng
import json
from scipy.constants import c
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

rng = default_rng()


#%% Define a function to estimate the PDF given a histogram
def estimatePDF(d, N=400):
    h,x = np.histogram(d,N)
    x = (x[:-1] + x[1:])/2
    h = h/(np.mean(np.diff(x))*len(d))
    return h,x

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
def velocity2aeuCovariance(X, 
                           off_nadir,
                           R_v = np.eye(3)*0.04,
                           N = 10000):
    
    """ Define the off-nadir directional cosine """
    v = np.cos(np.radians(off_nadir))
    
    """ Create an orbit around Venus """
    venSAR = orbit(planet=venus(), angleUnits="degrees")
    
    """ Generate random velocities """
    SVe = np.vstack((np.zeros((3,N)),
                     generateGaussian(R_v,N)))
    
    """ Compute the reference aueIe frame that is in error 
        from wrong velocity """
    orbitAngle, ascendingNode = venSAR.setFromStateVector(X)
    aaeu, _ = venSAR.computeAEU(orbitAngle, v)
    
    """ Compute the desired aeuI frame for each true velocity """
    daeu = np.zeros((3,3,N), dtype = np.double)
    for k in range(N):
        orbitAngle, ascendingNode = venSAR.setFromStateVector(X + SVe[:,k])
        daeu[:,:,k], _ = venSAR.computeAEU(orbitAngle, v)
        
    """ Compute the tilt, elevation and azimuth angles """
    AEU = aeuAngles(daeu.T.dot(aaeu))
    
    """ Return computed standard deviations """
    return AEU.dot(AEU.T)/N


#%% Timing error
def timing2aeuCovariance(X, off_nadir, dT=5, N=10000):
    
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
    aaeu = np.zeros((3,3,N), dtype = np.double)
    for k in range(N):
        aaeu[:,:,k], _ = venSAR.computeAEU(orbitAngle + dO[k], v)
        
    """ Compute the tilt, elevation and azimuth angles """
    AEU = aeuAngles(np.moveaxis((aaeu.T.dot(daeu)).T, -1, 0))
    
    """ Return computed standard deviations """
    return AEU.dot(AEU.T)/N
    #return np.sqrt(np.mean(AEU**2, axis=0))
    
#%%
def generateGaussian(R, N = 10000):
    m,n = R.shape
    
    """ Perform Cholesky decomposition """
    cD = np.linalg.cholesky(R)
    
    """ Generate the random data """
    return cD.dot(np.random.randn(m,N))

#%%
def aeu2rpyCovariance(R_AEU, e_ang = 14.28, N=100000):
    """ Compute the rotation matrix to go from aeu to ijk_s """
    ep_ang = np.radians(e_ang)
    c_ep = np.cos(ep_ang)
    s_ep = np.sin(ep_ang)
    M_er = np.array([
                    [1,     0,    0],
                    [0,  c_ep, s_ep],
                    [0, -s_ep, c_ep]
                   ])
    
    AEU = generateGaussian(R_AEU, N)
    RPY = np.zeros_like(AEU)
    RPYfromAEU(AEU, M_er, RPY)
    return RPY.dot(RPY)/N

#%%
def rpy2aeuCovariance(R_RPY, e_ang = 14.28, N=100000):
    """ Compute the rotation matrix to go from aeu to ijk_s """
    ep_ang = np.radians(e_ang)
    c_ep = np.cos(ep_ang)
    s_ep = np.sin(ep_ang)
    M_er = np.array([
                    [1,     0,    0],
                    [0,  c_ep, s_ep],
                    [0, -s_ep, c_ep]
                   ])
    
    RPY = generateGaussian(R_RPY, N)
    AEU = np.zeros_like(RPY)
    AEUfromRPY(RPY, M_er, AEU)
    return AEU.dot(AEU.T)/N

#%%
def contributors2aeuCovariance(X,
                               r,
                               off_nadir, 
                               e_ang = 14.28,
                               R_RPY = np.diag([3e-3, 0.4e-3, 3e-3])**2,
                               R_v = np.diag([0.2, 0.2, 0.2])**2,
                               R_t = 5**2,
                               R_p = 430**2,
                               N = 100000):
    """
    This function seeks to combine source errors into an aeu covariance matrix
    

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    off_nadir : TYPE
        DESCRIPTION.
    e_ang : TYPE
        DESCRIPTION.
    R_RPY : TYPE
        DESCRIPTION.
    R_v : TYPE
        DESCRIPTION.
    dT : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    R_att = rpy2aeuCovariance(R_RPY, e_ang, N)
    R_vel = velocity2aeuCovariance(X, off_nadir, R_v)
    R_tme = timing2aeuCovariance(X, off_nadir, R_t)
    R_pos = R_p/r**2
    
    R = np.array(block_diag(R_att, R_tme, R_vel, R_pos))
    AEU_m = np.array([[1,0,0,1,0,0,1,0,0,0],
                      [0,1,0,0,1,0,0,1,0,1],
                      [0,0,1,0,0,1,0,0,1,0]])
    return R, AEU_m
    
#%%
def simulateError(X, 
                  off_nadir,
                  R_RPY = np.diag([3e-3, 0.4e-3, 3e-3])**2,
                  R_v = np.diag([0.2, 0.2, 0.2])**2,
                  R_t = 5**2,
                  R_p = 430**2,
                  e_ang = 14.28,
                  azAxis = 6.0,
                  elAxis = 0.6,
                  carrier = 3.15e9,
                  n_AEU = 1000000,
                  loglevel = 0):
    
    """ Define a return dictionary """
    res = {"given": {"off_nadir": off_nadir,
                     "azAxis": azAxis,
                     "elAxis": elAxis,
                     "carrier": carrier,
                     "errorCovariances": {
                         "attitude": R_RPY.tolist(),
                         "velocity": R_v.tolist(),
                         "timing": R_t,
                         "position": R_p
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

    aeuI, aeuTCN = venSAR.computeAEU(orbitAngle, off_nadir)
    tcnI = aeuI.dot(aeuTCN.T)
    
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
    qq = aeuTCN.dot(M_e)
    # print(qq)
    # print(ijk_s)
    # rpyAnglesFromIJK((tcnI.T).dot(ijk_s).reshape((1,3,3)))
    # rpyAnglesFromIJK(ijk_s.reshape((1,3,3)))
    # theta_r, theta_p, theta_y = rpyAnglesFromIJK(ijk_s.reshape((1,3,3))).flatten()
    theta_r, theta_p, theta_y = rpyAnglesFromIJK(qq.reshape((1,3,3))).flatten()
    # (theta_y, theta_r, theta_p), _ = venSAR.YRPfromRotation(ijk_s)

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
        print("Roll: %0.4f, Pitch: %0.4f, Yaw: %0.4f" % (np.degrees(theta_r),
                                                         np.degrees(theta_p),
                                                         np.degrees(theta_y)))
        
    res["attitude"] = {"ijk_s": ijk_s.tolist(),
                       "tcnI": tcnI.tolist(),
                       "aeuI": aeuI.tolist(),
                       "aeuP": aeuP.tolist(),
                       "PRY": {"units": "degrees",
                               "roll": np.degrees(theta_r),
                               "pitch": np.degrees(theta_p),
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
    R, AEU_m = contributors2aeuCovariance(X,
                                          np.linalg.norm(R),
                                          off_nadir, 
                                          e_ang = e_ang,
                                          R_RPY = R_RPY,
                                          R_v = R_v,
                                          R_t = R_t,
                                          R_p = R_p)
    
    R_AEU = AEU_m.dot(R).dot(AEU_m.T)
    aeu_e = generateGaussian(R_AEU, n_AEU)
    aeu_m = np.zeros((3,3,n_AEU), dtype=aeu_e.dtype)
    aeu2rot(np.eye(3), aeu_e, aeu_m)
    aeuPe = np.einsum("ij,jkl->ikl", aeuP, aeu_m)
            
    """ Make a record of the AEU error covariances """
    res["computed"]["ErrorCovariance"] = {"units": "rad^2",
                                          "BlockCovariance": R.tolist(),
                                          "AEUCovariance": R_AEU.tolist()}
    
    """ Get the rotation matrices to steer boresight to edges of swath """
    eang = np.array([[0]*3 + [vAngles[0], 0.0, vAngles[-1]] + [0]*3]).reshape((3,3))
    erot = np.zeros((3,3,3), dtype = eang.dtype)
    aeu2rot(np.eye(3), np.radians(eang), erot)
    
    """ Calculate the zeroDoppler values """
    
    zDop = 2*np.einsum("i,ijk,jmn->kmn", XP[3:], aeuPe, erot)[:,2,:]/wavelength
    dc_max = zDop[range(n_AEU), np.argmax(np.abs(zDop), axis=1)]
    # alpha_list = sigmaA*rng.standard_normal(nA)
    # epsilon_list = sigmaE*rng.standard_normal(nE)
    # tau_list = sigmaT*rng.standard_normal(nT)
    
    # """ Generate the AEU vectors with the errors """
    # aeuPe = venSAR.pointingErrors(aeuP, alpha_list, epsilon_list, tau_list)
    
    # """ Compute the Doppler centroids for the min and max elevation angles """
    # dc = venSAR.dopCens(aeuPe, [vAngles[0], vAngles[-1]], XP[3:], wavelength)
    
    # """
    # This mapping of a function along the last axis allows computation of the
    # greatest difference of the Doppler centroid from zero. It isn't particularly
    # fast, but also not too slow."""
    # dc_max = np.max(np.abs(dc), axis=-1)

    """ Plot the histogram of the maximum Doppler centroids across the beam. """
    hD, xD = estimatePDF(dc_max, N=200)
    dError = 100*np.sum(np.abs(dc_max) > 1/15*dopBW)/n_AEU
    if loglevel > 2:
        print("Percent of Doppler centroids in violation of zero-Doppler threshold: %0.4f" % dError)
        
    if loglevel > 1:
        plt.figure()
        plt.plot(xD, hD)
        plt.axvline(x = 1/15*dopBW, color = 'r', label = '1/30 of Doppler Bandwdith')
        plt.axvline(x = -1/15*dopBW, color = 'r', label = '-1/30 of Doppler Bandwdith')
        plt.xlabel('Doppler centroid (Hz)')
        plt.ylabel('Histogram count')
        mytitle = r"Error Rate %0.2e" % (dError/100.0)
        plt.title(mytitle)
        plt.grid()
        plt.show()


    
    """ Compute the reference min and max ranges """
    uProt = np.einsum("ij,jlm->ilm", aeuP, erot)[:,2,:]
    refR = sv.computeRangeVectorsU(XP, uProt.T)
    refRng = np.linalg.norm(refR, axis=1)
    
    uPerot = np.einsum("ijk,jmn->imnk", aeuPe, erot)[:,2,:,:].T.reshape((3*n_AEU,3))
    errR = sv.computeRangeVectorsU(XP, uPerot)
    errRng = np.linalg.norm(errR, axis=-1).reshape((n_AEU,3))
    
    """ define vectorized functions to compute min and max over array """
    minofmax = np.min(np.stack((np.max(errRng, axis=-1), 
                                np.max(refRng)*np.ones(n_AEU))), axis=0)
    maxofmin = np.max(np.stack((np.min(errRng, axis=-1), 
                                np.min(refRng)*np.ones(n_AEU))), axis=0)
    coverage = (minofmax-maxofmin)/(np.max(refRng) - np.min(refRng))
    
    hS, xS = estimatePDF(coverage, N=50)
    sError = 100*np.sum(coverage < 14/15)/n_AEU
    if loglevel > 2:
        print("Percent of Swaths in violation of overlap threshold: %0.4f" % sError)
        
    if loglevel > 1:
        plt.figure()
        plt.plot(xS, hS)
        plt.axvline(x = 14/15, color = 'r', label = '14/15 of beamwidth')
        plt.xlabel('Fraction of elevation beam overlap on ground (unitless).')
        plt.ylabel('Histogram count')
        mytitle = r"Error Rate %0.2e" % (sError/100.0)
        plt.title(mytitle)
        plt.grid()
        plt.show()
    res["ErrorRate"] = {"Doppler": dError,
                        "Swath": sError}
    
    del aeuPe
    
    if loglevel > 0:
        print(json.dumps(res, indent=2))
    
    return res
# #%%
# def simulateError(X, 
#                   off_nadir,
#                   sigmaA = 0.023,
#                   sigmaE = 0.26,
#                   sigmaT = 0.12,
#                   sigmaP = 600,
#                   e_ang = 14.28,
#                   azAxis = 6.0,
#                   elAxis = 0.6,
#                   carrier = 3.15e9,
#                   nA = 300,
#                   nE = 300,
#                   nT = 100,
#                   loglevel = 0):
    
#     """ Define a return dictionary """
#     res = {"given": {"off_nadir": off_nadir,
#                      "azAxis": azAxis,
#                      "elAxis": elAxis,
#                      "carrier": carrier,
#                      "variances": {
#                          "sigmaA": sigmaA,
#                          "sigmaE": sigmaE,
#                          "sigmaT": sigmaT,
#                          "sigmaP": sigmaP
#                          }
#                      }
#            }
    
#     wavelength = c/carrier
#     elBW = wavelength/elAxis
#     azBW = wavelength/azAxis
    
#     res["computed"] = {"wavelength": wavelength,
#                        "beamwidths": {"units": "degrees",
#                                       "azimuth": np.degrees(azBW),
#                                       "elevation": np.degrees(elBW)}
#                        }
    
#     """ Create an orbit around Venus """
#     venSAR = orbit(planet=venus(), angleUnits="degrees")
#     orbitAngle, ascendingNode = venSAR.setFromStateVector(X)
    
#     """ Define the off-nadir angle """
#     v = np.cos(np.radians(off_nadir))
    
#     XI, rI, vI = venSAR.computeR(orbitAngle)
#     res["State Vector Radius"] = np.linalg.norm(X[:3])
#     res["Kepler"] = {"radius": np.linalg.norm(rI),
#                      "a": venSAR.a,
#                      "e": venSAR.e,
#                      "period": venSAR.period,
#                      "altitude": np.linalg.norm(rI) - venSAR.planet.a,
#                      "angles": {"units": "degrees",
#                                 "orbit": orbitAngle,
#                                 "ascending node": ascendingNode,
#                                 "inclination": np.degrees(venSAR.inclination),
#                                 "perigee": np.degrees(venSAR.arg_perigee)}
#                      }
#     e1, e2 = venSAR.computeE(orbitAngle, v)

#     aeuI, aeuTCN = venSAR.computeAEU(orbitAngle, v)
    
#     """ Compute the rotation matrix to go from aeu to ijk_s """
#     ep_ang = np.radians(e_ang)
#     c_ep = np.cos(ep_ang)
#     s_ep = np.sin(ep_ang)
#     M_e = np.array([
#                     [0,     1,     0],
#                     [s_ep,  0,  c_ep],
#                     [c_ep,  0, -s_ep]
#                    ])
#     ijk_s = aeuI.dot(M_e) 
#     rpyAnglesFromIJK(ijk_s.reshape((1,3,3)))
#     theta_r, theta_p, theta_y = rpyAnglesFromIJK(ijk_s.reshape((1,3,3))).flatten()
#     # (theta_y, theta_r, theta_p), _ = venSAR.YRPfromRotation(ijk_s)

#     M, dM = venSAR.computeItoR(orbitAngle)
#     XP = np.hstack((M.dot(XI[0:3]), dM.dot(XI[0:3]) + M.dot(XI[3:])))
#     vP = np.linalg.norm(XP[3:])
#     dopBW = 2*vP/wavelength*azBW
#     aeuP = M.dot(aeuI)
#     vAngles = np.arange(-np.degrees(elBW/2), np.degrees(elBW/2), 0.01)

#     if loglevel > 2:
#         print("Norm of u: %0.6f" % np.linalg.norm(aeuTCN[:,2]))
#         print("u*e1: %0.6f" % np.dot(aeuTCN[:,2],e1))
#         print("u*e2: %0.6f, v: %0.6f" % (np.dot(aeuTCN[:,2],e2), v))
#         print("uP*VP: %0.6f" % XP[3:].dot(aeuP[:,2]))
#         print("Roll: %0.4f, Pitch: %0.4f, Yaw: %0.4f" % (np.degrees(theta_r),
#                                                          np.degrees(theta_p),
#                                                          np.degrees(theta_y)))
        
#     res["attitude"] = {"ijk_s": ijk_s.tolist(),
#                        "aeuI": aeuI.tolist(),
#                        "aeuP": aeuP.tolist(),
#                        "PRY": {"units": "degrees",
#                                "roll": np.degrees(theta_r),
#                                "pitch": np.degrees(theta_p),
#                                "yaw": np.degrees(theta_y)}
#                        }

#     """ Compute the time """
#     t = venSAR.computeT(orbitAngle)
#     res["Kepler"]["time"] = t
        
#     """ Instantiate a state vector object """
#     sv = state_vector(planet=venus(), harmonicsCoeff=180)
    
#     """ Compute ECEF state vector """ 
#     XP = sv.toPCR(XI, t)
    
#     """ Compute the ground position """
#     XG = sv.computeGroundPositionU(XP, aeuP[:,2])
    
#     """ Test the ground position """
#     R = XG - XP[:3]
#     rhat = R/np.linalg.norm(R)
#     vG = -XP[:3].dot(rhat)/np.linalg.norm(XP[:3])
#     pG = (XG[0]/venSAR.planet.a)**2 + (XG[1]/venSAR.planet.a)**2 + (XG[2]/venSAR.planet.b)**2
#     dG = XP[3:].dot(rhat)
#     res["VCR"] = {"X": list(XG),
#                   "cosine off-nadir": vG,
#                   "Doppler Centroid": dG,
#                   "EllipsoidFn": pG}
    
#     """ Generate random errors """ 
#     R_AEU = contributors2aeuCovariance(X,
#                                        np.linalg.norm(R),
#                                        off_nadir, 
#                                        e_ang = e_ang,
#                                        R_RPY = np.diag([3e-3, 0.4e-3, 3e-3])**2,
#                                        R_v = np.diag([0.2, 0.2, 0.2])**2,
#                                        R_t = 5**2,
#                                        R_p = 430**2):
    
#     alpha_list = sigmaA*rng.standard_normal(nA)
#     epsilon_list = sigmaE*rng.standard_normal(nE)
#     tau_list = sigmaT*rng.standard_normal(nT)
    
#     """ Generate the AEU vectors with the errors """
#     aeuPe = venSAR.pointingErrors(aeuP, alpha_list, epsilon_list, tau_list)
    
#     """ Compute the Doppler centroids for the min and max elevation angles """
#     dc = venSAR.dopCens(aeuPe, [vAngles[0], vAngles[-1]], XP[3:], wavelength)
    
#     """
#     This mapping of a function along the last axis allows computation of the
#     greatest difference of the Doppler centroid from zero. It isn't particularly
#     fast, but also not too slow."""
#     dc_max = np.max(np.abs(dc), axis=-1)

#     """ Plot the histogram of the maximum Doppler centroids across the beam. """
#     N = np.prod(list(dc_max.shape))
#     hD, xD = estimatePDF(dc_max.reshape((N,)), N=200)
#     idxD = np.argwhere(xD > 1/15*dopBW)
#     idxD = idxD[0][0] if idxD.size > 0 else len(xD)
#     dError = 100*np.sum(hD[idxD:])*(xD[1]-xD[0])
#     if loglevel > 2:
#         print("Percent of Doppler centroids in violation of zero-Doppler threshold: %0.4f" % dError)
        
#     if loglevel > 1:
#         plt.figure()
#         plt.plot(xD, hD)
#         plt.axvline(x = 1/15*dopBW, color = 'r', label = '1/15 of Doppler Bandwdith')
#         plt.xlabel('Doppler centroid (Hz)')
#         plt.ylabel('Histogram count')
#         mytitle = r"Max $f_{dc}$, $\sigma_{\alpha}=%0.3f$, $\sigma_{\epsilon}=%0.3f$, $\sigma_{\tau}=%0.3f$ (deg)" % (sigmaA, sigmaE, sigmaT)
#         plt.title(mytitle)
#         plt.grid()
#         plt.show()


#     """ Compute beam projection differences """
#     off_boresight = [vAngles[0], 0, vAngles[-1]]
#     vS = np.array([[0, 
#                     np.sin(np.radians(ob)),
#                     np.cos(np.radians(ob))] for ob in off_boresight])
    
#     """ Compute the reference min and max ranges """
#     refR0 = sv.computeRangeVectorsU(XP, np.matmul(aeuP, vS[0,:]))
#     refR1 = sv.computeRangeVectorsU(XP, np.matmul(aeuP, vS[1,:]))
#     refR2 = sv.computeRangeVectorsU(XP, np.matmul(aeuP, vS[2,:]))
#     refRng0 = np.linalg.norm(refR0, axis=-1)
#     refRng1 = np.linalg.norm(refR1, axis=-1)
#     refRng2 = np.linalg.norm(refR2, axis=-1)
    
#     """ Compute the perturbed min and max ranges """
#     R0 = sv.computeRangeVectorsU(XP, np.matmul(aeuPe, vS[0,:]))
#     R1 = sv.computeRangeVectorsU(XP, np.matmul(aeuPe, vS[1,:]))
#     R2 = sv.computeRangeVectorsU(XP, np.matmul(aeuPe, vS[2,:]))
    
#     Rng0 = np.linalg.norm(R0, axis=-1)
#     Rng1 = np.linalg.norm(R1, axis=-1)
#     Rng2 = np.linalg.norm(R2, axis=-1)
    
#     """ define vectorized functions to compute min and max over array """
#     myfmin = np.vectorize(lambda x, y: x if x < y else y)
#     myfmax = np.vectorize(lambda x, y: x if x > y else y)
#     coverage = (myfmin(Rng2, refRng2) - myfmax(Rng0, refRng0))/(refRng2 - refRng0)
    
#     hS, xS = estimatePDF(coverage.reshape((N,)), N=50)
#     idxS = np.argwhere(xS < 14/15)[-1][0]
#     sError = 100*np.sum(hS[0:idxS])*(xS[1]-xS[0])
#     if loglevel > 2:
#         print("Percent of Swaths in violation of overlap threshold: %0.4f" % sError)
        
#     if loglevel > 1:
#         plt.figure()
#         plt.plot(xS, hS)
#         plt.axvline(x = 14/15, color = 'r', label = '14/15 of beamwidth')
#         plt.xlabel('Fraction of elevation beam overlap on ground (unitless).')
#         plt.ylabel('Histogram count')
#         mytitle = r"Swath overlap ratio, $\sigma_{\alpha}=%0.3f$, $\sigma_{\epsilon}=%0.3f$, $\sigma_{\tau}=%0.3f$ (deg)" % (sigmaA, sigmaE, sigmaT)
#         plt.title(mytitle)
#         plt.grid()
#         plt.show()
#     res["ErrorRate"] = {"Doppler": dError,
#                         "Swath": sError}
    
#     del aeuPe

#     """ Calculate error in epsilon due to orbit errors """
#     sigmaEs = np.degrees(sigmaP/np.linalg.norm(R))
    
#     alpha_list = sigmaA*rng.standard_normal(nA)
#     epsilon_list = (sigmaE - sigmaEs)*rng.standard_normal(nE)
#     tau_list = sigmaT*rng.standard_normal(nT)
    
#     """ Generate the ijk vectors with the errors in the SC frame """
#     ijk_se = venSAR.pointingErrors(aeuI, alpha_list, epsilon_list, tau_list).dot(M_e)
        
#     """ Flatten arrays so that we can use numba funtion """
#     YRP = np.zeros((N, 3), dtype=np.double)
#     YRPfromRotation(ijk_se.reshape((N,3,3)), ijk_s, YRP)
    
#     """ Calculate the reference roll pitch and roll """
#     (pitch_0, roll_0, yaw_0), _ = venSAR.YRPfromRotation(ijk_s)
    
#     """ Calculate the variances in YRP """
#     vAET = np.array([sigmaA, sigmaE - sigmaEs, sigmaT])/180*np.pi*1e3
#     vYRP = np.sqrt(np.mean(YRP**2, axis=0))*1e3
#     if loglevel > 2:
#         print("Standard deviation in Alpha, Epsilon, Tau:")
#         print(vAET)
#         print("Standard deviation in Yaw, Roll, Pitch:")
#         print(vYRP)
        
#     res["pointingErrors"] = {"units": "mrad",
#                              "AlphaEpsilonTau": list(vAET),
#                              "YawRollPitch": list(vYRP)}
    
#     res["computed"]["errors"] = {"sigmaDelta": sigmaEs}
    
#     if loglevel > 0:
#         print(json.dumps(res, indent=2))
    
#     return res