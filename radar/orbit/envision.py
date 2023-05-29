# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:30:18 2022

@author: ishuwa.sikaneta
"""
import os
import sys
import numpy as np
from measurement.measurement import state_vector
from space.planets import venus

#%% Define a default orbit file for Envision
if 'linux' in sys.platform:
    svfile = os.path.join(r"/users/isikanet/local/data/Orbits",
                          r"EnVision_ET1_2031_NorthVOI.oem")
else:
    svfile = os.path.join(r"C:\Users",
                          r"ishuwa.sikaneta",
                          r"OneDrive - ESA",
                          r"Documents",
                          r"ESTEC",
                          r"Envision",
                          r"Orbits",
                          r"EnVision_ET1_2031_NorthVOI.oem")

#%% Load an oem orbit state vector file from Venus
def loadSV(orbitfile = svfile, toPCR = False):
    """
    Load Envision State Vector data
    
    This function loads Envision state vector data from files
    provided by ESOC in the form of text *.oem files. It is
    assumed that the state vectors contained in this file
    correspond to the EME2000 inertial coordinate system and 
    that they need to be converted to a Venus-rotation-axis-related
    coordinate system.
    
    Assumptions
    -----------
    Data in the input file are in the EME2000 coordinate system. This
    assumption can be checked by reading the header of the file.

    Parameters
    ----------
    orbitfile : str, optional
        DESCRIPTION. The oem file to load. If not specified, the default
        defined in the svfile variable will be used.
    toPCR : bool, optional
        DESCRIPTION. Flag to specify that the state vectors should be
        converted to the IAU_VENUS reference frame.

    Returns
    -------
    sv : TYPE
        DESCRIPTION.

    """
    with open(svfile, 'r') as f:
        svecs = f.read().split('\n')
        
    svecs = [s.split() for s in svecs[16:-1]]
    svecs = [s for s in svecs if len(s)==7]
    svs = [(np.datetime64(s[0]), np.array([float(x) for x in s[1:]])*1e3) 
           for s in svecs]
    
    
    sv = state_vector(planet=venus())
    for vec in svs:
        mysv = sv.planet.ICRFtoPCR(*vec) if toPCR else sv.planet.ICRFtoPCI(*vec)
        sv.add(*mysv)
        
    # R_EMEVEN = sv.planet.EME_R
    # if toPCR:
    #     # w0 = sv.planet.w*((svs[0][0] - np.datetime64("2000-01-01T12:00"))
    #     #                   /np.timedelta64(1,'s'))
    #     w0 = sv.planet.wOffset(svs[0][0])#np.mod(w0, 2*np.pi)
    #     for vec in svs:
    #         """ Compute the time so we can transform to PCR """
    #         t = (vec[0] - svs[0][0])/np.timedelta64(1,'s')
            
    #         """ Add the state vector """
    #         sv.add(vec[0], sv.toPCR(R_EMEVEN.dot(vec[1]), t, w0))
    # else:
    #     for vec in svs:
    #         """ Add the state vector """
    #         sv.add(vec[0], R_EMEVEN.dot(vec[1]))
        
    return sv

#%% Old stuff

# class simulation:
#     def __init__(self, planet = earth()):
#         self.planet = planet
        
#     #%% Define a function to estimate the PDF given a histogram
#     def estimatePDF(self, d, N=400):
#         h,x = np.histogram(d,N)
#         x = (x[:-1] + x[1:])/2
#         h = h/(np.mean(np.diff(x))*len(d))
#         return h,x
    
#     #%%
#     def generateGaussian(self, R, N = 10000):
#         m,n = R.shape
        
#         """ Perform Cholesky decomposition """
#         cD = np.linalg.cholesky(R)
        
#         """ Generate the random data """
#         return cD.dot(np.random.randn(m,N))
    
#     #%% Define a function that will return state vectors in VCI coordinates
#     """ State vectors provided are in a coordinate system defined
#         by AOCS folks. The VCI coordinate system, as per Envision 
#         definitions, defines the j vector in the direction of the
#         ascending node. Thus, we need to find the components of the
#         state vectors in this coordinate system. """
#     def State(self, svs, idx):
#         planetOrbit = orbit(planet=self.planet, angleUnits="radians")
#         omega = np.array([planetOrbit.state2kepler(svs[k][1])["ascendingNode"] 
#                           for k in range(idx[0],idx[1])]).mean()
#         co = np.cos(omega)
#         so = np.sin(omega)
#         Mo = np.array([[co, so, 0],[-so,co,0],[0,0,1]])
#         Moo = np.block([[Mo, np.zeros_like(Mo)], [np.zeros_like(Mo), Mo]])
        
#         return [(svs[k][0], svs[k][1].dot(Moo)) for k in range(idx[0],idx[1])]


#     #%% angular error from velocity error
#     def velocity2aeuCovariance(self,
#                                X, 
#                                off_nadir,
#                                R_v = np.eye(3)*0.04,
#                                N = 10000):
        
#         """ Define the off-nadir directional cosine """
#         v = np.cos(np.radians(off_nadir))
        
#         """ Create an orbit around Venus """
#         planetOrbit = orbit(planet=self.planet, angleUnits="degrees")
        
#         """ Generate random velocities """
#         SVe = np.vstack((np.zeros((3,N)),
#                          self.generateGaussian(R_v,N)))
        
#         """ Compute the reference aueIe frame that is in error 
#             from wrong velocity """
#         orbitAngle, ascendingNode = planetOrbit.setFromStateVector(X)
#         aaeu, _ = planetOrbit.computeAEU(orbitAngle, v)
        
#         """ Compute the desired aeuI frame for each true velocity """
#         daeu = np.zeros((3,3,N), dtype = np.double)
#         for k in range(N):
#             orbitAngle, ascendingNode = planetOrbit.setFromStateVector(X + SVe[:,k])
#             daeu[:,:,k], _ = planetOrbit.computeAEU(orbitAngle, v)
            
#         """ Compute the tilt, elevation and azimuth angles """
#         AEU = aeuAngles(daeu.T.dot(aaeu))
        
#         """ Return computed standard deviations """
#         return AEU.dot(AEU.T)/N


#     #%% Timing error
#     def timing2aeuCovariance(self, X, off_nadir, dT=5, N=10000):
        
#         """ Define the off-nadir directional cosine """
#         v = np.cos(np.radians(off_nadir))
        
#         """ Create an orbit around Venus """
#         planetOrbit = orbit(planet=self.planet, angleUnits="degrees")
        
#         """ Compute the orbit angle for the given state vector"""
#         orbitAngle, ascendingNode = planetOrbit.setFromStateVector(X)
        
#         """ Compute constant relating derivative of orbit angle to derivative 
#             of time """
#         e = planetOrbit.e
#         GM = planetOrbit.planet.GM
#         a = planetOrbit.a
#         U = np.radians(orbitAngle) - planetOrbit.arg_perigee
#         C = (1+e*np.cos(U))**2*np.sqrt(GM/
#                              (a*(1-e**2))**3)
        
#         """ Generate random orbit angle deviances """
#         dO = rng.standard_normal(N)*dT*C
        
#         """ Calculate the DAEU frame """
#         daeu, _ = planetOrbit.computeAEU(orbitAngle, v)
        
#         """ Loop over deviations in orbit angle """
#         aaeu = np.zeros((3,3,N), dtype = np.double)
#         for k in range(N):
#             aaeu[:,:,k], _ = planetOrbit.computeAEU(orbitAngle + dO[k], v)
            
#         """ Compute the tilt, elevation and azimuth angles """
#         AEU = aeuAngles(np.moveaxis((aaeu.T.dot(daeu)).T, -1, 0))
        
#         """ Return computed standard deviations """
#         return AEU.dot(AEU.T)/N


#     #%%
#     def aeu2rpyCovariance(self, R_AEU, e_ang = 14.28, N=100000):
#         """ Compute the rotation matrix to go from aeu to ijk_s """
#         ep_ang = np.radians(e_ang)
#         c_ep = np.cos(ep_ang)
#         s_ep = np.sin(ep_ang)
#         M_er = np.array([
#                         [1,     0,    0],
#                         [0,  c_ep, s_ep],
#                         [0, -s_ep, c_ep]
#                        ])
        
#         AEU = self.generateGaussian(R_AEU, N)
#         RPY = np.zeros_like(AEU)
#         RPYfromAEU(AEU, M_er, RPY)
#         return RPY.dot(RPY)/N

#     #%%
#     def rpy2aeuCovariance(self, R_RPY, e_ang = 14.28, N=100000):
#         """ Compute the rotation matrix to go from aeu to ijk_s """
#         ep_ang = np.radians(e_ang)
#         c_ep = np.cos(ep_ang)
#         s_ep = np.sin(ep_ang)
#         M_er = np.array([
#                         [1,     0,    0],
#                         [0,  c_ep, s_ep],
#                         [0, -s_ep, c_ep]
#                        ])
        
#         RPY = self.generateGaussian(R_RPY, N)
#         AEU = np.zeros_like(RPY)
#         AEUfromRPY(RPY, M_er, AEU)
#         return AEU.dot(AEU.T)/N

#     #%%
#     def contributors2aeuCovariance(self,
#                                    X,
#                                    r,
#                                    off_nadir, 
#                                    e_ang = 14.28,
#                                    R_RPY = np.diag([3e-3, 0.4e-3, 3e-3])**2,
#                                    R_v = np.diag([0.2, 0.2, 0.2])**2,
#                                    R_t = 5**2,
#                                    R_p = 430**2,
#                                    N = 100000):
#         """
#         This function seeks to combine source errors into an aeu covariance matrix
        
    
#         Parameters
#         ----------
#         X : TYPE
#             DESCRIPTION.
#         off_nadir : TYPE
#             DESCRIPTION.
#         e_ang : TYPE
#             DESCRIPTION.
#         R_RPY : TYPE
#             DESCRIPTION.
#         R_v : TYPE
#             DESCRIPTION.
#         dT : TYPE
#             DESCRIPTION.
    
#         Returns
#         -------
#         None.
    
#         """
        
#         R_att = self.rpy2aeuCovariance(R_RPY, e_ang, N)
#         R_vel = self.velocity2aeuCovariance(X, off_nadir, R_v)
#         R_tme = self.timing2aeuCovariance(X, off_nadir, R_t)
#         R_pos = R_p/r**2
        
#         R = np.array(block_diag(R_att, R_tme, R_vel, R_pos))
#         AEU_m = np.array([[1,0,0,1,0,0,1,0,0,0],
#                           [0,1,0,0,1,0,0,1,0,1],
#                           [0,0,1,0,0,1,0,0,1,0]])
#         return R, AEU_m
    
#     #%%
#     def simulateError(self,
#                       X, 
#                       off_nadir,
#                       R_RPY = np.diag([3e-3, 0.4e-3, 3e-3])**2,
#                       R_v = np.diag([0.2, 0.2, 0.2])**2,
#                       R_t = 5**2,
#                       R_p = 430**2,
#                       e_ang = 14.28,
#                       azAxis = 6.0,
#                       elAxis = 0.6,
#                       carrier = 3.15e9,
#                       n_AEU = 1000000,
#                       loglevel = 0
#                       ):
        
#         """ Define a return dictionary """
#         res = {"given": {"off_nadir": off_nadir,
#                          "azAxis": azAxis,
#                          "elAxis": elAxis,
#                          "carrier": carrier,
#                          "errorCovariances": {
#                              "attitude": R_RPY.tolist(),
#                              "velocity": R_v.tolist(),
#                              "timing": R_t,
#                              "position": R_p
#                              }
#                          }
#                }
        
#         wavelength = c/carrier
#         elBW = wavelength/elAxis
#         azBW = wavelength/azAxis
        
#         res["computed"] = {"wavelength": wavelength,
#                            "beamwidths": {"units": "degrees",
#                                           "azimuth": np.degrees(azBW),
#                                           "elevation": np.degrees(elBW)}
#                            }
        
#         """ Create an orbit around Venus """
#         planetOrbit = orbit(planet=self.planet, angleUnits="degrees")
#         orbitAngle, ascendingNode = planetOrbit.setFromStateVector(X)
        
#         """ Define the off-nadir angle """
#         v = np.cos(np.radians(off_nadir))
        
#         XI, rI, vI = planetOrbit.computeR(orbitAngle)
#         res["State Vector Radius"] = np.linalg.norm(X[:3])
#         res["Kepler"] = {"radius": np.linalg.norm(rI),
#                          "a": planetOrbit.a,
#                          "e": planetOrbit.e,
#                          "period": planetOrbit.period,
#                          "altitude": np.linalg.norm(rI) - planetOrbit.planet.a,
#                          "angles": {"units": "degrees",
#                                     "orbit": orbitAngle,
#                                     "ascending node": ascendingNode,
#                                     "inclination": np.degrees(planetOrbit.inclination),
#                                     "perigee": np.degrees(planetOrbit.arg_perigee)}
#                          }
#         e1, e2 = planetOrbit.computeE(orbitAngle, v)
    
#         aeuI, aeuTCN = planetOrbit.computeAEU(orbitAngle, off_nadir)
#         tcnI = aeuI.dot(aeuTCN.T)
        
#         """ Compute the rotation matrix to go from aeu to ijk_s """
#         ep_ang = np.radians(e_ang)
#         c_ep = np.cos(ep_ang)
#         s_ep = np.sin(ep_ang)
#         M_e = np.array([
#                         [0,     1,     0],
#                         [s_ep,  0,  c_ep],
#                         [c_ep,  0, -s_ep]
#                        ])
#         ijk_s = aeuI.dot(M_e) 
#         qq = aeuTCN.dot(M_e)
    
#         """ Compute the Euler angles """
#         theta_r, theta_p, theta_y = rpyAnglesFromIJK(qq.reshape((1,3,3))).flatten()
    
#         M, dM = planetOrbit.computeItoR(orbitAngle)
#         XP = np.hstack((M.dot(XI[0:3]), dM.dot(XI[0:3]) + M.dot(XI[3:])))
#         vP = np.linalg.norm(XP[3:])
#         dopBW = 2*vP/wavelength*azBW
#         aeuP = M.dot(aeuI)
#         vAngles = np.arange(-np.degrees(elBW/2), np.degrees(elBW/2), 0.01)
    
#         if loglevel > 2:
#             print("Norm of u: %0.6f" % np.linalg.norm(aeuTCN[:,2]))
#             print("u*e1: %0.6f" % np.dot(aeuTCN[:,2],e1))
#             print("u*e2: %0.6f, v: %0.6f" % (np.dot(aeuTCN[:,2],e2), v))
#             print("uP*VP: %0.6f" % XP[3:].dot(aeuP[:,2]))
#             print("Roll: %0.4f, Pitch: %0.4f, Yaw: %0.4f" % (np.degrees(theta_r),
#                                                              np.degrees(theta_p),
#                                                              np.degrees(theta_y)))
            
#         res["attitude"] = {"ijk_s": ijk_s.tolist(),
#                            "tcnI": tcnI.tolist(),
#                            "aeuI": aeuI.tolist(),
#                            "aeuP": aeuP.tolist(),
#                            "PRY": {"units": "degrees",
#                                    "roll": np.degrees(theta_r),
#                                    "pitch": np.degrees(theta_p),
#                                    "yaw": np.degrees(theta_y)}
#                            }
    
#         """ Compute the time """
#         t = planetOrbit.computeT(orbitAngle)
#         res["Kepler"]["time"] = t
            
#         """ Instantiate a state vector object """
#         sv = state_vector(planet=self.planet, harmonicsCoeff=180)
        
#         """ Compute ECEF state vector """ 
#         XP = sv.toPCR(XI, t)
        
#         """ Compute the ground position """
#         XG = sv.computeGroundPositionU(XP, aeuP[:,2])
        
#         """ Test the ground position """
#         R = XG - XP[:3]
#         rhat = R/np.linalg.norm(R)
#         vG = -XP[:3].dot(rhat)/np.linalg.norm(XP[:3])
#         pG = (XG[0]/planetOrbit.planet.a)**2 + (XG[1]/planetOrbit.planet.a)**2 + (XG[2]/planetOrbit.planet.b)**2
#         dG = XP[3:].dot(rhat)
#         res["VCR"] = {"X": list(XG),
#                       "cosine off-nadir": vG,
#                       "Doppler Centroid": dG,
#                       "EllipsoidFn": pG}
        
#         """ Generate random errors """ 
#         R, AEU_m = self.contributors2aeuCovariance(X,
#                                                    np.linalg.norm(R),
#                                                    off_nadir, 
#                                                    e_ang = e_ang,
#                                                    R_RPY = R_RPY,
#                                                    R_v = R_v,
#                                                    R_t = R_t,
#                                                    R_p = R_p)
        
#         R_AEU = AEU_m.dot(R).dot(AEU_m.T)
#         aeu_e = self.generateGaussian(R_AEU, n_AEU)
#         aeu_m = np.zeros((3,3,n_AEU), dtype=aeu_e.dtype)
#         aeu2rot(np.eye(3), aeu_e, aeu_m)
#         aeuPe = np.einsum("ij,jkl->ikl", aeuP, aeu_m)
                
#         """ Make a record of the AEU error covariances """
#         res["computed"]["ErrorCovariance"] = {"units": "rad^2",
#                                               "BlockCovariance": R.tolist(),
#                                               "AEUCovariance": R_AEU.tolist()}
        
#         """ Get the rotation matrices to steer boresight to edges of swath """
#         eang = np.array([[0]*3 + [vAngles[0], 0.0, vAngles[-1]] + [0]*3]).reshape((3,3))
#         erot = np.zeros((3,3,3), dtype = eang.dtype)
#         aeu2rot(np.eye(3), np.radians(eang), erot)
        
#         """ Calculate the zeroDoppler values """
#         zDop = 2*np.einsum("i,ijk,jmn->kmn", XP[3:], aeuPe, erot)[:,2,:]/wavelength
#         dc_max = zDop[range(n_AEU), np.argmax(np.abs(zDop), axis=1)]
    
#         """ Plot the histogram of the maximum Doppler centroids across the beam. """
#         hD, xD = self.estimatePDF(dc_max, N=200)
#         dError = 100*np.sum(np.abs(dc_max) > 1/15*dopBW)/n_AEU
#         if loglevel > 2:
#             print("Percent of Doppler centroids in violation of zero-Doppler threshold: %0.4f" % dError)
            
#         if loglevel > 1:
#             plt.figure()
#             plt.plot(xD, hD)
#             plt.axvline(x = 1/15*dopBW, color = 'r', label = '1/30 of Doppler Bandwdith')
#             plt.axvline(x = -1/15*dopBW, color = 'r', label = '-1/30 of Doppler Bandwdith')
#             plt.xlabel('Doppler centroid (Hz)')
#             plt.ylabel('Histogram count')
#             mytitle = r"Error Rate %0.2e" % (dError/100.0)
#             plt.title(mytitle)
#             plt.grid()
#             plt.show()
    
    
        
#         """ Compute the reference min and max ranges """
#         uProt = np.einsum("ij,jlm->ilm", aeuP, erot)[:,2,:]
#         refR = sv.computeRangeVectorsU(XP, uProt.T)
#         refRng = np.linalg.norm(refR, axis=1)
        
#         uPerot = np.einsum("ijk,jmn->imnk", aeuPe, erot)[:,2,:,:].T.reshape((3*n_AEU,3))
#         errR = sv.computeRangeVectorsU(XP, uPerot)
#         errRng = np.linalg.norm(errR, axis=-1).reshape((n_AEU,3))
        
#         """ define vectorized functions to compute min and max over array """
#         minofmax = np.min(np.stack((np.max(errRng, axis=-1), 
#                                     np.max(refRng)*np.ones(n_AEU))), axis=0)
#         maxofmin = np.max(np.stack((np.min(errRng, axis=-1), 
#                                     np.min(refRng)*np.ones(n_AEU))), axis=0)
#         coverage = (minofmax-maxofmin)/(np.max(refRng) - np.min(refRng))
        
#         hS, xS = self.estimatePDF(coverage, N=50)
#         sError = 100*np.sum(coverage < 14/15)/n_AEU
#         if loglevel > 2:
#             print("Percent of Swaths in violation of overlap threshold: %0.4f" % sError)
            
#         if loglevel > 1:
#             plt.figure()
#             plt.plot(xS, hS)
#             plt.axvline(x = 14/15, color = 'r', label = '14/15 of beamwidth')
#             plt.xlabel('Fraction of elevation beam overlap on ground (unitless).')
#             plt.ylabel('Histogram count')
#             mytitle = r"Error Rate %0.2e" % (sError/100.0)
#             plt.title(mytitle)
#             plt.grid()
#             plt.show()
#         res["ErrorRate"] = {"Doppler": dError,
#                             "Swath": sError}
        
#         del aeuPe
        
#         if loglevel > 0:
#             print(json.dumps(res, indent=2))
        
#         return res
