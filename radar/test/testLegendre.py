#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 15:59:04 2020

@author: ishuwa
"""


from timeit import default_timer as timer
import numpy as np
from measurement.measurement import state_vector

#%%
sv = state_vector()

#
# start = timer()
# for x in np.arange(0,np.pi/2,np.pi/200):
#     dummy = sv.myLegendre_old(360, np.sin(x))
# end = timer()
# print(end - start)

# start = timer()
# for x in np.arange(0,np.pi/2,np.pi/100):
#     dummy = sv.myLegendre(360, np.sin(x))
# end = timer()
# print(end - start)

#%%
mydata = np.array([-5.28682880e+05, -6.12367342e+06,  3.49575263e+06,  1.41881891e+03,
        -3.79246352e+03, -6.42885957e+03])
mytime = [np.datetime64('2015-01-01T00:00:04.721666194')]

#%%
start = timer()
eState = sv.expandedState(mydata, 0.0)
end = timer()
print(end - start)
print(eState)