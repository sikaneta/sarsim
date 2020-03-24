#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 21:25:24 2020

@author: ishuwa
"""
mysim = "/home/ishuwa/simulation/15cm/15cm_simulation.xml"
radar = cfg.loadConfiguration(mysim)

#%%
m,n = data.shape
osf = 4
#%%
fdata = np.fft.fft(data, axis=0)

#%%
fs = cfg.FFT_freq(m,m,0).astype(int)
ofs = cfg.FFT_freq(m*osf,m*osf,0).astype(int)
fdata = np.zeros((m*osf,n), dtype=np.complex128)
fdata[fs,:] = np.fft.fft(data,axis=0)

#%% Define which bands to look at
bands = np.arange(-int(len(radar)/2)-2,int(len(radar)/2)+1+2)

#%% Get an r_sys object
r_sys = cfg.radar_system(radar, bands)

#%%
s = np.arange(m*osf)/(r_sys.ksp*r_sys.n_bands*osf)
s -= np.mean(s) + 0.23537 #49747 #0*0.0721/osf
#%%
nddata = np.fft.ifft(np.exp(-0.482*osf*1j*2*np.pi*np.matlib.repmat(ofs, n, 1).T/4096)*fdata, axis=0)

#%%
nddata = nddata/np.max(np.abs(nddata))
plt.figure()
plt.plot(s,20*np.log10(np.abs(nddata[:,400])), 
         s,20*np.log10(np.abs(nddata[:,400])), '.')
plt.grid()
plt.xlabel("Arclength (m)")
plt.ylabel("Response (dB)")
plt.title("Azimuth impulse response")
plt.show()