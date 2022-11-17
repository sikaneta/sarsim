# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:28:27 2022

@author: ishuwa.sikaneta
"""

import configuration.configuration as cfg
import numpy as np
import os
import utils.fileio as fio
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%%
config_xml = r"C:\Users\ishuwa.sikaneta\local\data\simulation\1m_mode.xml"
cargs = r"--config-xml %s" % config_xml
ridx = [0, None]

#%% Load the radar configuration
radar = cfg.loadConfiguration(config_xml)

#%% Compute the ground point and associate slow time parameters
r_sys = cfg.loadRsys(radar)

#%% Load the raw data
data = fio.loadNumpy_raw_dataMem(radar, ridx = ridx)[0,:,:]

#%% shift the data with fftshift
data = np.fft.ifft(data, axis=1)

# #%%
# data = np.fft.fftshift(data, axes=1)
#%%
data = data/np.max(np.abs(data[:]))

#%%
folder = r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\Presentations\SAR Course"
filename = os.path.join(folder, "compressed_range.png")
#%% Plot the data
s_skip = 25
extent = [r_sys.C.s[0]/s_skip, 
          r_sys.C.s[-1]/s_skip, 
          r_sys.r[-1],
          r_sys.r[0]]

fig, ax = plt.subplots()

plt.imshow(np.abs(data[:,0:-1:s_skip])**2, 
           cmap=mpl.colormaps["Blues"],
           extent = extent)

plt.ylabel('Range (m)')
plt.xlabel('Azimuth (%d m)' % s_skip)

divider = make_axes_locatable(ax)
cax1 = divider.append_axes("right", size="5%", pad=0.05)

cbar = plt.colorbar(cax = cax1)

#plt.savefig(filename, transparent=True, dpi=300, bbox_inches="tight")
plt.show()

#%%
az_signal_pwr = 20*np.log10(np.sum(np.abs(data), axis=0))
az_signal_pwr -= np.max(az_signal_pwr)
az_signal_phs = np.unwrap(np.angle(np.sum(data, axis=0)))
az_signal_phs -= np.max(az_signal_phs)

#%%
plt.figure()
m,n = data.shape
plt.plot(20*np.log10(np.abs(data[:,int(n/2)])))
plt.grid()
plt.show()

#%%
filename = os.path.join(folder, "signalAmpPhs.png")
fig, ax1 = plt.subplots()

ax1.plot(r_sys.C.s/1e3, az_signal_phs)
ax1.set_title('Signal phase and amplitude')
ax1.set_ylabel('Phase (rad)', color='C0')
ax1.tick_params(axis='y', color='C0', labelcolor='C0')
ax1.spines['right'].set_color('C0')
ax1.set_xlabel('Along track position (Km)')

ax2 = ax1.twinx()
ax2.plot(r_sys.C.s/1e3, az_signal_pwr,'C1')
ax2.set_ylim(-50,5)
ax2.set_ylabel('Response (dB)', color='C1')
ax2.tick_params(axis='y', color='C1', labelcolor='C1')
ax2.spines['right'].set_color('C1')
ax2.set_xlabel('Along track position (Km)')
plt.grid()

plt.savefig(filename, transparent=True, dpi=300, bbox_inches="tight")

plt.show()
