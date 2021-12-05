# -*- coding: utf-8 -*-

import os
import configuration.configuration as cfg
import numpy as np
import matplotlib.pyplot as plt
import argparse

#%% Define the default configuration file
defaultConfig = os.path.join(os.environ["HOMEPATH"], 
                             "local", 
                             "sarsim", 
                             "radar", 
                             "generateXML", 
                             "sureConfig.xml")

#%% Load the config file
radar = cfg.loadConfiguration(defaultConfig)

#%% Load some angles to plot
aLength = (np.mean(radar[0]['antenna']['azimuthLengths'])
           *len(radar[0]['antenna']['azimuthLengths']))

bWidth = radar[0]['antenna']['wavelength']/aLength

angles = np.arange(-2*bWidth, 2*bWidth, 0.001)

#%% Calculate the two-way pattern
pattern = cfg.twoWayArrayPattern(angles, radar[0])

#%% Plot the result
plt.figure()
plt.plot(angles, np.abs(pattern))
plt.grid()
plt.show()

#%% Make a polar plot
ampfactor = 25
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.fill(ampfactor*angles, 
        np.abs(pattern), 
        alpha=0.2, 
        facecolor="yellow")
ax.fill(ampfactor*angles, 
        np.abs(pattern), 
        facecolor="none", 
        edgecolor='red',
        linewidth=1)
ax.set_rticks([])
ax.grid(False)
ax.axis("off")
plt.savefig('C:/Users/Ishuwa.Sikaneta/local/sarsim/radar/test/pattern.svg')
# plt.show()

