# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 09:55:47 2021

@author: Ishuwa.Sikaneta
"""

from measurement.measurement import measurement
from measurement.measurement import state_vector
import xml.etree.ElementTree as etree
import numpy as np
import quaternion
from space.planets import earth

#%%
defXML = r"C:\Users\Ishuwa.Sikaneta\local\Data\Sentinel-1\S1A_IW_SLC__1SDV_20211003T173327_20211003T173354_039958_04BAA1_314F.SAFE\annotation\s1a-iw1-slc-vv-20211003t173327-20211003t173353-039958-04baa1-004.xml"

#%%
class attitude(measurement):
    def __init__(self, atFile = None):
        if atFile is not None:
            self.readMeasurements(atFile)
            
    def readMeasurements(atFile):
        pass
    
    def estimate(self, dtime):
        minK = self.findNearest(dtime)
        y = self.measurementData[minK]
        return y    
#%%   
class attitude_S1(attitude):
    """
    Class to load Sentinel-1 attitude
    
    Methods
    -------
    readAttitude:
        Reads state vectors from an Sarscape style XML file.
        
    """
    def readMeasurements(self, XMLFile):
        dataPool = etree.parse(XMLFile).getroot()
        aElementList = dataPool.find(".//attitudeList")
        aElements = aElementList.findall(".//attitude")
        qElements = ["q3", "q0", "q1", "q2"]
        wElements = ["wx", "wy", "wz"]
        euler = ["roll", "pitch", "yaw"]
        
        self.measurementTime = [np.datetime64(a.find("time").text) 
                                for a in aElements]
        
        self.qData = [np.quaternion(*[float(a.find(q).text) for q in qElements]) 
                      for a in aElements]
        
        self.wData = [np.array([float(a.find(w).text) for w in wElements]) 
                      for a in aElements]
        
        self.eData = [np.array([float(a.find(e).text) for e in euler]) 
                      for a in aElements]

        return

#%%
class state_vector_S1(state_vector):
    """
    Class to load Sarscape format state vectors
    
    Methods
    -------
    readStateVectors:
        Reads state vectors from an Sarscape style XML file.
        
    """
    def readStateVectors(self, SMLFile):
        dataPool = etree.parse(SMLFile).getroot()
        orbits = dataPool.findall('.//orbit')
        sElements = ['x', 'y', 'z']
        for orbit in orbits:
            t = np.datetime64(orbit.find("time").text)
            pos = orbit.find('position')
            X = [float(pos.find(f).text) for f in sElements]
            vel = orbit.find('velocity')
            V = [float(vel.find(f).text) for f in sElements]
            self.add(t, X + V)
            
        return
    
#%% Coded from https://sentinels.copernicus.eu/documents/247904/1877131/Sentinel-1-Level-1-Detailed-Algorithm-Definition.pdf/22ab8853-214c-481f-a24f-33a1d471de58?t=1561759089000    
def attcalc(X, roll, pitch, yaw, basis = np.array([1,0,0])):
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    
    # R = np.array([[1, 0,  0  ],
    #               [0, cr, -sr],
    #               [0, sr, cr ]])
    
    # P = np.array([[cp,  0,  sp],
    #               [0,   1,  0 ],
    #               [-sp, 0,  cp]])
    
    # Y = np.array([[cy, -sy, 0],
    #               [sy, cy,  0],
    #               [0,  0,   1]])
    
        
    R = np.array([[1, 0,  0  ],
                  [0, cr, -sr],
                  [0, sr, cr ]])
    
    P = np.array([[cp,  0,  sp],
                  [0,   1,  0 ],
                  [-sp, 0,  cp]])
    
    Y = np.array([[cy, -sy, 0],
                  [sy, cy,  0],
                  [0,  0,   1]])
    
    # basis = np.array([1,0,0])
    # offset = np.radians(-7)
    # basis = np.array([np.cos(offset),0,-np.sin(offset)])
    # Y.dot(P).dot(R).dot(basis)
    
    planet = earth()
    
    q = X[0:3]*np.array([1,1,1/(1-planet.e**2)])
    c = q/np.linalg.norm(q)
    
    w = np.array([0, 0, planet.w])
    vi = X[3:] + np.cross(w, X[0:3])
    
    b = np.cross(c, vi)
    b /= np.linalg.norm(b)
    
    a = np.cross(b, c)
    
    L = np.stack((a, b, c), axis=1)
    
    return L.dot(Y).dot(P).dot(R).dot(basis), L, R, P, Y

#%% Read some data
sv = state_vector_S1(defXML)
at = attitude_S1(defXML)

#%%
X = sv.estimate(at.measurementTime[0])
roll, pitch, yaw = at.eData[0]
    
