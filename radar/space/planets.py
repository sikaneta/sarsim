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
    nharmonics = 360
    
    """ 
    The following values, to relate to J2000 EME2000 from:
    https://web.archive.org/web/20111024101856/http://astrogeology.usgs.gov/Projects/WGCCRE/constants/iau2000_table1.html
    """
    alpha = 0.0
    delta = 90.0
    w0 = 190.147
    
    
    def __init__(self):
        self.e = np.sqrt(1 - (self.b/self.a)**2)
        self.nbody = {"name": [],
                      "GM": [],
                      "position": np.array([])}
        self.EME_R = self.ICRF2Planet()
    
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
    
    def wOffset(self, 
                target_time, 
                ref_time = np.datetime64("2000-01-01T12:00")):
        w1 = self.w*((target_time - ref_time)
                     /np.timedelta64(1,'s'))
        return np.mod(w1 + np.radians(self.w0), 2*np.pi) 
        
    def ICRF2Planet(self):
        c_alpha = np.cos(np.radians(self.alpha))
        s_alpha = np.sin(np.radians(self.alpha))
        s_delta = np.sin(np.radians(self.delta))
        c_delta = np.cos(np.radians(self.delta))
        
        R = np.array([[-s_alpha, -c_alpha*s_delta, c_alpha*c_delta],
                      [c_alpha, -s_alpha*s_delta, s_alpha*c_delta],
                      [0, c_delta, s_delta]]).T
        
        return np.block([[R, np.zeros_like(R)],[np.zeros_like(R), R]])
        
        
        
            
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
    nharmonics = 360
        
    """ 
    The following values, to relate to J2000 EME2000 from:
    https://web.archive.org/web/20111024101856/http://astrogeology.usgs.gov/Projects/WGCCRE/constants/iau2000_table1.html
    """
    alpha = 0.0
    delta = 90.0
    w0 = 190.147
    
class venus(solar_planet):
    M = 3.24858592079e14/G
    GM = 3.24858592079e14
    a = 6051878.0
    b = 6051878.0
    siderealSeconds = 243.0226*24*60*60
    #w = -2*np.pi/(243.0226*24*60*60)
    """ Changed in favour of IAU specs for Venus """
    w = -np.radians(1.4813688/24/60/60)
    J2 = 4.4044e-6
    J4 = -2.1474e-6
    J6 = 0
    J8 = 0
    sphericalHarmonicsFile = os.path.join(harmonicsPath , "shgj180u.txt")
    nharmonics = 180
    
    """ 
    The following values, to relate to J2000 EME2000 from: 
    https://web.archive.org/web/20111024101856/http://astrogeology.usgs.gov/Projects/WGCCRE/constants/iau2000_table1.html
    """
    alpha = 272.76
    delta = 67.16
    w0 = 160.2
    