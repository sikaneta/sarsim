#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 16:06:54 2021

@author: ishuwa
"""

from datetime import datetime
import numpy as np
from astropy import coordinates as coord
from astropy import units as u
from astropy.time import Time


#sun_acc = GM_sun.to_value()*(sun_pos/np.linalg.norm(sun_pos)**3 
#                             - (sun_pos-sat_pos)/np.linalg.norm(sun_pos-sat_pos)**3)

#%% Define the nbody class
class nbody:
    wE = 7.292115e-5
    def __init__(self):
        pass
    
    def sun(self, my_date):
        mytime = Time(my_date)
        gcrs=coord.get_sun(mytime)
        itrs = gcrs.transform_to(coord.ITRS(obstime=mytime,
                                            representation_type='cartesian'))
        loc = coord.EarthLocation(*itrs.cartesian.xyz)
        print(loc.lat, loc.lon, loc.height)
        return loc.to(u.m).to_value()
    
    
    def moon(self, my_date):
        mytime = Time(my_date)
        gcrs=coord.get_moon(mytime)
        itrs = gcrs.transform_to(coord.ITRS(obstime=mytime,
                                            representation_type='cartesian'))
        loc = coord.EarthLocation(*itrs.cartesian.xyz)
        print(loc.lat, loc.lon, loc.height)
        return loc.to(u.m).to_value()