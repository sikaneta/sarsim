# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:10:22 2022

@author: Ishuwa.Sikaneta
"""

from space.planets import earth, venus
from orbit.orientation import orbit
import numpy as np
import matplotlib.pyplot as plt
from measurement.measurement import state_vector
from scipy.constants import c
from numpy.random import default_rng
rng = default_rng()

sentinel = False

#%% Test creation of orbit object with radians as input
if sentinel:
    off_nadir = 30
    orbit_angle = 79.7417 + 33
    v = np.cos(np.radians(off_nadir))
    
    sentinel = orbit(0.0001303, 
                     np.radians(79.7417), 
                     7167100, 
                     np.radians(98.1813))
    XI, rI, vI = sentinel.computeR(np.radians(orbit_angle))
    
    print("|r|: %0.6f, r: %0.6f" % (np.linalg.norm(XI[0:3]), rI))
    print("|v|: %0.6f, v: %0.6f" % (np.linalg.norm(XI[3:]), vI))
    
    aeuI, aeuTCN = sentinel.computeAEU(np.radians(orbit_angle), v)
    e1, e2 = sentinel.computeE(np.radians(orbit_angle), v)
    M, dM = sentinel.computeItoR(np.radians(orbit_angle))
    XP = np.hstack((M.dot(XI[0:3]), dM.dot(XI[0:3]) + M.dot(XI[3:])))
    uP = M.dot(aeuI[:,2])
    
    print("Norm of u: %0.6f" % np.linalg.norm(aeuTCN[:,2]))
    print("u*e1: %0.6f" % np.dot(aeuTCN[:,2],e1))
    print("u*e2: %0.6f, v: %0.6f" % (np.dot(aeuTCN[:,2],e2), np.cos(np.radians(off_nadir))))
    print("uP*VP: %0.6f" % XP[3:].dot(uP))

#%% Test computation of time from orbit angle
if sentinel:
    beta = np.arange(0,720,0.01)
    t = sentinel.computeT(np.radians(beta))
    print(max(t)-min(t))
    print(2*np.pi*np.sqrt(sentinel.a**3/sentinel.planet.GM))
    print("-"*40)

#%% Test obrit object with degreees as input
if sentinel:
    sentinel = orbit(0.0001303, 
                     79.7417, 
                     7167100, 
                     98.1813,
                     angleUnitFunction=np.radians)
    
    XI, rI, vI = sentinel.computeR(orbit_angle)
    
    print("|r|: %0.6f, r: %0.6f" % (np.linalg.norm(XI[0:3]), rI))
    print("|v|: %0.6f, v: %0.6f" % (np.linalg.norm(XI[3:]), vI))
    
    aeuI, aeuTCN = sentinel.computeAEU(orbit_angle, v)
    e1, e2 = sentinel.computeE(orbit_angle, v)
    M, dM = sentinel.computeItoR(orbit_angle)
    XP = np.hstack((M.dot(XI[0:3]), dM.dot(XI[0:3]) + M.dot(XI[3:])))
    aeuP = M.dot(aeuI)
    print("Norm of u: %0.6f" % np.linalg.norm(aeuTCN[:,2]))
    print("u*e1: %0.6f" % np.dot(aeuTCN[:,2],e1))
    print("u*e2: %0.6f, v: %0.6f" % (np.dot(aeuTCN[:,2],e2), 
                                     np.cos(np.radians(off_nadir))))
    print("uP*VP: %0.6f" % XP[3:].dot(aeuP[:,2]))

#%% Test computation of orbit angle from time
if sentinel:
    b,e = sentinel.computeO(10)
    print("Angle: %0.6f, Error: %0.7f" % (b,e))

#%% Load an oem orbit state vecctor file from Venus
svfile = r"C:\Users\Ishuwa.Sikaneta\Documents\ESTEC\Envision\EnVision_ALT_T4_2032_SouthVOI.oem"
with open(svfile, 'r') as f:
    svecs = f.read().split('\n')
svecs = [s.split() for s in svecs[16:-1]]
svecs = [s for s in svecs if len(s)==7]
svs = [(np.datetime64(s[0]), np.array([float(x) for x in s[1:]])*1e3) 
       for s in svecs]

#%% Test converting to Kepler orbit elements
venSAR = orbit(planet=venus())
sidx_left = 0
sidx_right = 1000
kepler = [venSAR.state2kepler(s[1]) for s in svs[sidx_left:sidx_right]] 
t = [(s[0] - svs[0][0])/np.timedelta64(24,'h') for s in svs[sidx_left:sidx_right]]
ts = [(s[0] - svs[0][0])/np.timedelta64(1,'s') for s in svs[sidx_left:sidx_right]]

#%%
xidx = 100
yidx = 248
sv = state_vector(planet=venus(), harmonicsCoeff=180)
sv.add(svs[xidx][0], sv.toPCR(svs[xidx][1], ts[xidx]))
dt = (svs[yidx][0] - svs[xidx][0])/np.timedelta64(1,'s')
print(dt)   
print(sv.estimate(svs[yidx][0]) - sv.toPCR(svs[yidx][1], ts[xidx]+dt))

venSAR.setFromStateVector(svs[xidx][1])
print(venSAR.period)

dT = sv.measurementTime[0] + np.timedelta64(int(venSAR.period*1e9), 'ns')
Y = sv.estimate(dT)
dX = sv.measurementData[-1]-sv.measurementData[0]
print("Difference in state vectors after 1 period")
print("="*40)
print(dX)

#%% Plot some figures from the previous
plt.figure()
plt.plot(t,[np.degrees(k["inclination"]) for k in kepler])
plt.grid()
plt.title('Inclination Angle')
plt.ylabel('Angle (degrees)')
plt.xlabel('Time (s)')
plt.show()

plt.figure()
plt.plot(t,[np.degrees(k["ascendingNode"]) for k in kepler])
plt.grid()
plt.title('Angle of Ascending Node')
plt.ylabel('Angle (degrees)')
plt.xlabel('Time (s)')
plt.show()

plt.figure()
plt.plot(t,[np.degrees(k["perigee"]) for k in kepler])
plt.grid()
plt.title('Angle of Perigee')
plt.ylabel('Angle (degrees)')
plt.xlabel('Time (s)')
plt.show()

plt.figure()
plt.plot(t,[k["eccentricity"] for k in kepler])
plt.grid()
plt.title('Eccentricity')
plt.ylabel('Eccentricity (unitless)')
plt.xlabel('Time (s)')
plt.show()

#%% Calculate some parameters for venSAR
wavelength = c/3.15e9
elAxis = 0.6
azAxis = 6.0

elBW = wavelength/elAxis
azBW = wavelength/azAxis
print("Elevation beamwidth (deg): %0.4f" % np.degrees(elBW))
print("Azimuth beamwidth (deg): %0.4f" % np.degrees(azBW))

#%% Test zero-Doppler steering for Venus orbit
orbit_angle = np.radians(130)
v = np.cos(np.radians(30))


venSAR = orbit(kepler[xidx]["eccentricity"],
               kepler[xidx]["perigee"],
               kepler[xidx]["a"],
               kepler[xidx]["inclination"],
               planet=venus())

XI, rI, vI = venSAR.computeR(orbit_angle)
e1, e2 = venSAR.computeE(orbit_angle, v)

aeuI, aeuTCN = venSAR.computeAEU(orbit_angle, v)
M, dM = venSAR.computeItoR(orbit_angle)
XP = np.hstack((M.dot(XI[0:3]), dM.dot(XI[0:3]) + M.dot(XI[3:])))
aeuP = M.dot(aeuI)

print("Norm of u: %0.6f" % np.linalg.norm(aeuTCN[:,2]))
print("u*e1: %0.6f" % np.dot(aeuTCN[:,2],e1))
print("u*e2: %0.6f, v: %0.6f" % (np.dot(aeuTCN[:,2],e2), v))
print("uP*VP: %0.6f" % XP[3:].dot(aeuP[:,2]))

#%%  Test zero-Doppler steering for Venus orbit for a different orbit angle
orbit_angle = 79.7417
v = np.cos(np.radians(41.2))

venSAR = orbit(kepler[xidx]["eccentricity"],
               np.degrees(kepler[xidx]["perigee"]),
               kepler[xidx]["a"],
               np.degrees(kepler[xidx]["inclination"]),
               planet=venus(),
               angleUnitFunction=np.radians)

XI, rI, vI = venSAR.computeR(orbit_angle)
e1, e2 = venSAR.computeE(orbit_angle, v)

aeuI, aeuTCN = venSAR.computeAEU(orbit_angle, v)

M, dM = venSAR.computeItoR(orbit_angle)
XP = np.hstack((M.dot(XI[0:3]), dM.dot(XI[0:3]) + M.dot(XI[3:])))
aeuP = M.dot(aeuI)

print("Norm of u: %0.6f" % np.linalg.norm(aeuTCN[:,2]))
print("u*e1: %0.6f" % np.dot(aeuTCN[:,2],e1))
print("u*e2: %0.6f, v: %0.6f" % (np.dot(aeuTCN[:,2],e2), v))
print("uP*VP: %0.6f" % XP[3:].dot(aeuP[:,2]))

#%% Check that rotating the look vector around the azimuth axis is okay
vAngles = np.arange(-np.degrees(elBW/2), np.degrees(elBW/2), 0.01)
dummy = venSAR.rotateUnit(vAngles, aeuP[:,2], aeuP[:,0])
plt.figure()
plt.plot(vAngles, 2*dummy.dot(XP[3:])/wavelength)
plt.title('Doppler centroid as a function of elevation angle')
plt.xlabel('Elevation angle (deg)')
plt.ylabel('Doppler centroid (Hz)')
plt.grid()
plt.show()

#%% Plot Doppler centroid after first introducing a tilt angle
aeuPe = venSAR.pointingError(aeuP, -0.0, 0.0, 0.1)
dummy = venSAR.rotateUnit(vAngles, aeuPe[:,2], aeuPe[:,0])
plt.figure()
plt.plot(vAngles, 2*dummy.dot(XP[3:])/wavelength)
plt.grid()
plt.title('Doppler centroid as a function of elevation angle (with tilt angle error)')
plt.xlabel('Beam elevation angle (degrees)')
plt.ylabel('Doppler centroid (Hz)')
plt.show()

#%% Randomly generate errors and compute resulting Doppler centroid
""" Define the number of random elements for each of azimuth, elevation and
    tilt errors """
nA = 1000
nE = 100
nT = 100

""" Define the standard deviations of the different errors """
sigmaA = 0.01
sigmaE = 0.45
sigmaT = 0.15

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
# dc_max = np.apply_along_axis(lambda x: x[np.argmax(np.abs(x))], -1, dc)
dc_max = np.max(np.abs(dc), axis=-1)

""" Plot the histogram of the maximum Doppler centroids across the beam. """
N = np.prod(list(dc_max.shape))
plt.figure()
h = plt.hist(dc_max.reshape((N,)), 200)
plt.xlabel('Doppler centroid (Hz)')
plt.ylabel('Histogram count')
mytitle = r"Max $f_{dc}$, $\sigma_{\alpha}=%0.2f$, $\sigma_{\epsilon}=%0.2f$, $\sigma_{\tau}=%0.2f$ (deg)" % (sigmaA, sigmaE, sigmaT)
plt.title(mytitle)
plt.grid()
plt.show()

#%% Compute the ground positions of the center of the beam and the near and
#   far edges of the beam (In elevation)
""" First get the state vector """
sidx = 100
X = svs[sidx][1]

""" Compute the orbit angle """
orbitElements = venSAR.state2kepler(X)
orbitAngle = (orbitElements["trueAnomaly"] - 
              (orbitElements["ascendingNode"] 
               - orbitElements["perigee"]))

""" Compute the time """
t = venSAR.computeT(orbitAngle)

""" Compute ECEF state vector """ 
XP = sv.toPCR(X, t)

""" Compute the ground position """
XG = sv.computeGroundPositionU(XP, aeuP[:,2])
#XG = sv.computeGroundPosition(np.array(XP), 0.1, v, 0)

#%% Print some test results of the ground position
""" Test the ground position """
R = XG - XP[:3]
rhat = R/np.linalg.norm(R)
vG = -XP[:3].dot(rhat)/np.linalg.norm(XP[:3])
pG = (XG[0]/venSAR.planet.a)**2 + (XG[1]/venSAR.planet.a)**2 + (XG[2]/venSAR.planet.b)**2
dG = XP[3:].dot(rhat)
print("Cosine of depression angle: %0.8f" % vG)
print("Doppler centroid: %0.8f" % dG)
print("Ellipsoid function value: %0.8f" % pG)

#%% Compute beam projection differences
off_boresight = [vAngles[0], 0, vAngles[-1]]
v = np.array([[0, 
              -np.sin(np.radians(ob)),
              np.cos(np.radians(ob))] for ob in off_boresight])

""" Compute the reference min and max ranges """
refR0 = sv.computeRangeVectorsU(XP, np.matmul(aeuP, v[0,:]))
refR1 = sv.computeRangeVectorsU(XP, np.matmul(aeuP, v[1,:]))
refR2 = sv.computeRangeVectorsU(XP, np.matmul(aeuP, v[2,:]))
refRng0 = np.sqrt(np.sum(refR0*refR0, -1))
refRng1 = np.sqrt(np.sum(refR1*refR1, -1))
refRng2 = np.sqrt(np.sum(refR2*refR2, -1))

""" Compute the perturbed min and max ranges """
R0 = sv.computeRangeVectorsU(XP, np.matmul(aeuPe, v[0,:]))
R1 = sv.computeRangeVectorsU(XP, np.matmul(aeuPe, v[1,:]))
R2 = sv.computeRangeVectorsU(XP, np.matmul(aeuPe, v[2,:]))

Rng0 = np.sqrt(np.sum(R0*R0, -1))



#%% Code to compute the ground position given u and v
# def computeGroundPosition(X, u = 0, v = np.pi/4, h=0):
#     xhat = -X[:3]/np.linalg.norm(X[:3])
#     vhat = X[3:]/np.linalg.norm(X[3:])
    
#     """ Compute a vector perpendicular to both """
#     what = np.cross(xhat, vhat)
#     what = what/np.linalg.norm(what)
    
#     """ Find the matrix to invert """
#     M = np.array([xhat, vhat, what])
#     Minv = np.linalg.inv(M)
    
#     """ Homogeneous solution """
#     y = np.array([v, u, 0])
#     uhom = np.dot(Minv, y)
    
#     """ Find a unit vector specific solution """
#     C = uhom.dot(uhom) - 1
#     B = 2*uhom.dot(what)
#     tA = (-B + np.sqrt(B**2 - 4*C))/2
#     tB = (-B - np.sqrt(B**2 - 4*C))/2
#     uhatA = uhom + tA*what
#     uhatB = uhom + tB*what
#     uhat = uhatA if uhatA.dot(what) >= 0 else uhatB
    
#     """ Now find where the vector hits the ground """
#     Xs = X[:3]
#     a = planet.a
#     b = planet.b
#     mXs = Xs/np.array([a+h, a+h, b+h])
#     mu = uhat/np.array([a+h, a+h, b+h])
    
#     mC = mXs.dot(mXs) - 1
#     mB = 2*mXs.dot(mu)
#     mA = mu.dot(mu)
    
#     mtA = (-mB + np.sqrt(mB**2 - 4*mA*mC))/(2*mA)
#     mtB = (-mB - np.sqrt(mB**2 - 4*mA*mC))/(2*mA)
    
#     mt = mtA if np.abs(mtA) < np.abs(mtB) else mtB
    
#     return Xs + mt*uhat
    