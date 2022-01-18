# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 09:58:29 2021

@author: Ishuwa.Sikaneta
"""

import os
import share.resource as res
harmonicsPath = os.path.split(res.__file__)[0]
import numpy as np
from scipy.constants import G

from astropy.time import Time
from astropy.coordinates import get_body
from astropy import coordinates as coord

masses = {
    "Earth": 5.972167867791379e+24,
    "Sun": 1.988409870698051e+30,
    "Jupiter": 1.8981245973360505e+27,
    "Moon": 7.34767309e+22}

class solar_planet:
    """
    Class to store constants for planet Earth
    
    This class stores constants such as mass, ellipticity, the first
    spherical harmonics and a function to load spherical harmonics from
    file.
    
    Methods
    -------
    
    loadHarmonics
        This function loads the spherical harmonics from a text file
    
    """
    M = masses["Earth"]
    GM = 3.986005e14
    a = 6378137.0
    b = 6356752.3141
    siderealSeconds = 86164.099
    w = 7.292115e-5
    J2 = 0.00108263
    J4 = -0.00000237091222
    J6 = 0.00000000608347
    J8 = -0.00000000001427
    sphericalHarmonicsFile = os.path.join(harmonicsPath , "egm96.txt")
    
    def __init__(self):
        self.e = np.sqrt(1 - (self.b/self.a)**2)
        self.nbody = {"name": [],
                      "GM": [],
                      "position": np.array([])}
    
    def set_nbody(self, t, names=[]):
        """
        Calculate the position of the list of bodies in names relative to this
        planet

        Parameters
        ----------
        t : `datetime.datetime`, `numpy.datetime64`, `astropy.Time`, 
            The UTC reference time at which to compute the positions of the 
            bodies
        names : list, optional
            The names of celestial bodies. These bodies should be known to 
            astropy. The default is [].

        Returns
        -------
        None.

        """
        
        bodies = np.array([get_body(name, Time(t))
                           .transform_to(coord.ITRS())
                           .cartesian.xyz.to_value(unit="m") 
                           for name in names]).T
        
        self.reference_time = t
        
        self.nbody = {"name": [name for name in names],
                      "GM": np.array([G*masses[name] for name in names]),
                      "position": bodies} 
        
        return
    
    def get_nbody(self, t):
        """
        Return coordinates of nbodies
        
        Return the coordinates of the nbodies defined in set_nbody as
        a function of number of seconds, t from the reference time for
        each of the nbodies

        Parameters
        ----------
        t : float
            The number of seconds from the reference time.

        Returns
        -------
        the coordinates of the nbodies.

        """
        
        R = np.array([[np.cos(self.w*t), np.sin(self.w*t), 0],
                      [-np.sin(self.w*t), np.cos(self.w*t), 0],
                      [0, 0, 1]])
        
        return R.dot(self.nbody["position"])
    
    def nbodyacc(self, t, X):
        """
        Nbody acceleration

        Parameters
        ----------
        t : `float`
            The number of seconds from the reference time.
        X : `np.ndarray[3]`
            The coordinates of the reference point at which to compute the
            acceleration.

        Returns
        -------
        `np.ndarray[3]`
            The components of the computed acceleration

        """
        
        # Don't do anything if no bodies have been defined
        if self.nbody['position'].size == 0:
            return np.array([0,0,0])
        
        # Otherwise, calculate accelerations
        R = np.array([[np.cos(self.w*t), -np.sin(self.w*t), 0],
                      [np.sin(self.w*t), np.cos(self.w*t), 0],
                      [0, 0, 1]])
        

        GMvec = self.nbody["GM"]
        positions = R.dot(self.nbody["position"])
        nm_positions = np.linalg.norm(positions, axis=0)
        
        m,n = positions.shape
        spositions = positions - np.tile(X[0:3], (n,1)).T
        nm_spositions = np.linalg.norm(spositions, axis=0)
        
        t1 = positions.dot(np.diag(GMvec/nm_positions**3))
        t2 = spositions.dot(np.diag(GMvec/nm_spositions**3))
        
        return np.sum(t1-t2, axis=1)
    
    def dragacc(self, 
                X,
                ballistic = 0.01,
                p1 = 2.0e-13,
                h1 = 5.0e5):
        h = np.linalg.norm(X[0:3]) - self.a
        H = 70000 + 0.075*(h - h1)
        p = p1*np.exp((h-h1)/H)
        V = X[3:6]
        return -p/2*ballistic*V*np.linalg.norm(V)
        
        
        
            
#%% Define planet Earth. The default
class earth(solar_planet):
    M = masses["Earth"]
    GM = 3.986005e14
    a = 6378137.0
    b = 6356752.3141
    siderealSeconds = 86164.099
    w = 7.292115e-5
    J2 = 0.00108263
    J4 = -0.00000237091222
    J6 = 0.00000000608347
    J8 = -0.00000000001427
    sphericalHarmonicsFile = os.path.join(harmonicsPath , "egm96.txt")
    
# earth = planet()

# #%% Define planet Venus.
# venus = planet()
# venus.GM = 4.8675e24*G
# venus.a = 6051800
# venus.b = 6051800
# venus.J2 = 4.458e-6
# venus.J4 = 0
# venus.J6 = 0
# venus.J8 = 0
# venus.siderealSeconds = (243.01*60*60*24)
# venus.w = 2*np.pi/venus.siderealSeconds