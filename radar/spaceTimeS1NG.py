# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:49:56 2023

@author: ishuwa.sikaneta
"""
#%%
import numpy as np
from configuration.configuration import plotSpaceTime

#%% parameters for S1NG
satvel = 7597.0
prf = 1237 # Swath 1 IW
#prf = 1501 # Swath 2 IW
#prf = 1053 # Stripmap
L = 12.28
nChan = 9
dx = L/nChan/2
radarS1NG = [{
     "platform": {
         "satelliteVelocity": satvel
     },
     "acquisition": {
         "prf": prf
     },
     "delay": {
         "sampling": 0,
         "baseline": k*dx
     }
} for k in np.arange(nChan) - int(nChan/2)]

plotSpaceTime(15, radarS1NG)

#%% parameters for RoseL
satvel = 7597.0
prf = 1378
#prf = 1489
#prf = 1406
L = 11
nChan = 5
dx = L/nChan/2
radarRoseL = [{
     "platform": {
         "satelliteVelocity": satvel
     },
     "acquisition": {
         "prf": prf
     },
     "delay": {
         "sampling": 0,
         "baseline": k*dx
     }
} for k in np.arange(nChan) - int(nChan/2)]

plotSpaceTime(10, radarRoseL)
