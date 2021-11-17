# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 09:58:29 2021

@author: Ishuwa.Sikaneta
"""

import os
import share.resource as res
harmonicsPath = os.path.split(res.__file__)[0]

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
    M = 5.97219e24
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