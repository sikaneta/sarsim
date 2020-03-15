# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:22:53 2019

@author: SIKANETAI
"""
import multichannel.msar as msar

#%%
rdidx0 = 0
rdidx1 = 7
idx0 = 6000 #5600
idx1 = 8000 #8000
x_axis = np.arange(idx1-idx0) + idx0
#%%
#df0 = np.load("E:\\Users\\SIKANETAI\\simulation\\30cm\\l0_data\\30cm_simulation_rx_c0b0.npy")
#df1 = np.load("E:\\Users\\SIKANETAI\\simulation\\30cm\\l0_data\\30cm_simulation_rx_c1b0.npy")
df0 = msar.computeRadarSignalGPU(radar[rdidx0], pointXYZ, satXYZ)
##%%
#plt.figure()
#plt.plot(x_axis, np.abs(np.sum(df0[:,idx0:idx1], axis=0)))
#plt.show()
#%%
df1 = msar.computeRadarSignalGPU(radar[rdidx1], pointXYZ, satXYZ)

#%%
angd = np.unwrap(np.angle(np.sum(df0[:,idx0:idx1], axis=0)*np.conj(np.sum(df1[:,idx0:idx1], axis=0))))
plt.figure()
plt.plot(angd)
plt.grid()
plt.show()

#%%
plt.figure()
plt.plot(x_axis, np.abs(np.sum(df0[:,idx0:idx1], axis=0)),
         x_axis, np.abs(np.sum(df1[:,idx0:idx1], axis=0)))
plt.show()

#%%
plt.figure()
plt.plot(np.abs(np.sum(df0[:,idx0:idx1], axis=0)*np.conj(np.sum(df1[:,idx0:idx1], axis=0))))
plt.show()

#%%
zp0 = msar.computeRadarSignal(radar[rdidx0], pointXYZ, satXYZ, 2048)
zp1 = msar.computeRadarSignal(radar[rdidx1], pointXYZ, satXYZ, 2048)

#%%
angz = np.unwrap(np.angle(np.sum(zp0[:,idx0:idx1], axis=0)*np.conj(np.sum(zp1[:,idx0:idx1], axis=0))))
plt.figure()
plt.plot(angz)
plt.grid()
plt.show()

#%%
plt.figure()
plt.plot(x_axis, np.abs(np.sum(zp0[:,idx0:idx1], axis=0)),
         x_axis, np.abs(np.sum(zp1[:,idx0:idx1], axis=0)))
plt.show()

#%%
plt.figure()
plt.plot(np.abs(np.sum(zp0[:,idx0:idx1], axis=0)*np.conj(np.sum(zp1[:,idx0:idx1], axis=0))))
plt.show()

#%%
plt.figure()
plt.plot(x_axis, np.mod(angd, 2*np.pi),
         x_axis, np.mod(angz, 2*np.pi))
plt.grid()
plt.show()


