# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 09:58:29 2021

@author: Ishuwa.Sikaneta
"""

import os
import share.resource as res
harmonicsPath = os.path.split(res.__file__)[0]
import numpy as np

from astropy.time import Time
from astropy.coordinates import get_body
from astropy import coordinates as coord

masses = {
    "Earth": 5.972167867791379e+24,
    "Sun": 1.988409870698051e+30,
    "Jupiter": 1.8981245973360505e+27,
    "Moon": 7.34767309e+22}

class earth:
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
    
    def nbody(t, names=[]):
        """
        Calculate the position of the list of bodies in names relative to this
        planet

        Parameters
        ----------
        t : `datetime.dateimte`, `numpy.datetime64`, `astropy.Time`, 
            The UTC time at which to compute the positions of the bodies
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
                           for name in names])
        
        return [masses[name] for name in names], bodies
        
            