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

#%% Test creation of orbit object with radians as input
off_nadir = 30
orbit_angle = 79.7417 + 33
v = np.cos(np.radians(off_nadir))

sentinel = orbit(0.0001303, 
                 np.radians(79.7417), 
                 7167100, 
                 np.radians(98.1813))
XI, rI, vI = sentinel.computeR(np.radians(orbit_angle))
# print(XI)

print("|r|: %0.6f, r: %0.6f" % (np.linalg.norm(XI[0:3]), rI))
print("|v|: %0.6f, v: %0.6f" % (np.linalg.norm(XI[3:]), vI))

aeuI, aeuTCN = sentinel.computeAEU(np.radians(orbit_angle), v)
e1, e2 = sentinel.computeE(np.radians(orbit_angle), v)
M, dM = sentinel.computeItoR(np.radians(orbit_angle))
# print(M)
# print(dM)
XP = np.hstack((M.dot(XI[0:3]), dM.dot(XI[0:3]) + M.dot(XI[3:])))
# print(XP)
uP = M.dot(aeuI[:,2])
print("Norm of u: %0.6f" % np.linalg.norm(aeuTCN[:,2]))
print("u*e1: %0.6f" % np.dot(aeuTCN[:,2],e1))
print("u*e2: %0.6f, v: %0.6f" % (np.dot(aeuTCN[:,2],e2), np.cos(np.radians(off_nadir))))
print("uP*VP: %0.6f" % XP[3:].dot(uP))

#%% Test computation of time from orbit angle
beta = np.arange(0,720,0.01)
t = sentinel.computeT(np.radians(beta))
print(max(t)-min(t))
print(2*np.pi*np.sqrt(sentinel.a**3/sentinel.planet.GM))
print("-"*40)

#%% Test obrit object with degreees as input
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
print("u*e2: %0.6f, v: %0.6f" % (np.dot(aeuTCN[:,2],e2), np.cos(np.radians(off_nadir))))
print("uP*VP: %0.6f" % XP[3:].dot(aeuP[:,2]))

#%% Test computation of orbit angle from time
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
kepler = [venSAR.state2kepler(s[1]) for s in svs] 
t = [(s[0] - svs[0][0])/np.timedelta64(24,'h') for s in svs]
ts = [(s[0] - svs[0][0])/np.timedelta64(1,'s') for s in svs]

#%%
xidx = 1000
yidx = 1248
sv = state_vector(planet=venus(), harmonicsCoeff=180)
sv.add(svs[xidx][0], sv.toECEF(svs[xidx][1], ts[xidx]))
dt = (svs[yidx][0] - svs[xidx][0])/np.timedelta64(1,'s')
print(dt)   
print(sv.estimate(svs[yidx][0]) - sv.toECEF(svs[yidx][1], ts[xidx]+dt))

venSAR.setFromStateVector(svs[xidx][1])
print(venSAR.period)

#%% Plot some figures from the previous
plt.figure()
plt.plot(t,[np.degrees(k["inclination"]) for k in kepler])
plt.grid()
plt.title('Inclination Angle')
plt.show()

plt.figure()
plt.plot(t,[np.degrees(k["ascendingNode"]) for k in kepler])
plt.grid()
plt.title('Angle of Ascending Node')
plt.show()

plt.figure()
plt.plot(t,[np.degrees(k["perigee"]) for k in kepler])
plt.grid()
plt.title('Angle of Perigee')
plt.show()

plt.figure()
plt.plot(t,[k["eccentricity"] for k in kepler])
plt.grid()
plt.title('Eccentricity')
plt.show()

#%% Test the impact of errors on the requirments svecs definedoff_nadir = 30
wavelength = c/3.15e9
orbit_angle = 19.7417
v = np.cos(np.radians(30))

venSAR = orbit(kepler[xidx]["eccentricity"],
               np.degrees(kepler[xidx]["perigee"]),
               kepler[xidx]["a"],
               np.degrees(kepler[xidx]["inclination"]),
               planet=venus(),
               angleUnitFunction=np.radians)

#venSAR = orbit(planet=venus(), angleUnitFunction=np.radians)
#venSAR.setFromStateVector(svs[xidx][1])
XI, rI, vI = venSAR.computeR(orbit_angle)
e1, e2 = venSAR.computeE(orbit_angle, v)

aeuI, aeuTCN = venSAR.computeAEU(orbit_angle, v)
M, dM = venSAR.computeItoR(orbit_angle)
XP = np.hstack((M.dot(XI[0:3]), dM.dot(XI[0:3]) + M.dot(XI[3:])))
aeuP = M.dot(aeuI)

print("Norm of u: %0.6f" % np.linalg.norm(aeuTCN[:,2]))
print("u*e1: %0.6f" % np.dot(aeuTCN[:,2],e1))
print("u*e2: %0.6f, v: %0.6f" % (np.dot(aeuTCN[:,2],e2), np.cos(np.radians(off_nadir))))
print("uP*VP: %0.6f" % XP[3:].dot(aeuP[:,2]))

#%% Test the impact of errors on the requirments svecs definedoff_nadir = 30
wavelength = c/3.15e9
orbit_angle = np.radians(130)
v = np.cos(np.radians(30))


venSAR = orbit(kepler[xidx]["eccentricity"],
               kepler[xidx]["perigee"],
               kepler[xidx]["a"],
               kepler[xidx]["inclination"],
               planet=venus())

#venSAR = orbit(planet=venus(), angleUnitFunction=np.radians)
#venSAR.setFromStateVector(svs[xidx][1])
XI, rI, vI = venSAR.computeR(orbit_angle)
e1, e2 = venSAR.computeE(orbit_angle, v)

aeuI, aeuTCN = venSAR.computeAEU(orbit_angle, v)
M, dM = venSAR.computeItoR(orbit_angle)
XP = np.hstack((M.dot(XI[0:3]), dM.dot(XI[0:3]) + M.dot(XI[3:])))
aeuP = M.dot(aeuI)

print("Norm of u: %0.6f" % np.linalg.norm(aeuTCN[:,2]))
print("u*e1: %0.6f" % np.dot(aeuTCN[:,2],e1))
print("u*e2: %0.6f, v: %0.6f" % (np.dot(aeuTCN[:,2],e2), np.cos(np.radians(off_nadir))))
print("uP*VP: %0.6f" % XP[3:].dot(aeuP[:,2]))
#%% Check that rotating th elook vector around the azimuth axis is okay
vAngles = np.arange(-5, 5, 0.01)
dummy = venSAR.rotateUnit(vAngles, aeuP[:,2], aeuP[:,0])
plt.figure()
plt.plot(vAngles, 2*dummy.dot(XP[3:])/wavelength)
plt.grid()
plt.show()

#%%
aeuPe = venSAR.pointingError(aeuP, 0.1, 0.0, 0.0)
dummy = venSAR.rotateUnit(vAngles, aeuPe[:,2], aeuPe[:,0])
plt.figure()
plt.plot(vAngles, 2*dummy.dot(XP[3:])/wavelength)
plt.grid()
plt.show()