# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:37:50 2022

@author: ishuwa.sikaneta
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams["figure.figsize"] = [7.00, 5.50]
plt.rcParams["figure.autolayout"] = True
#%%
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel("range (m)")
ax.set_ylabel("azimuth (m)")

plot = [ax.plot_surface(S, R, Z, cmap='viridis', rstride=1, cstride=1)]

#%%
ax.view_init(elev=30, azim=0)

#%%
def change_plot(frame_number, plot):
   #plot[0].remove()
   #plot[0] = ax.plot_surface(S, R, Z, cmap='viridis', rstride=1, cstride=1)
   #ax.plot_surface(S, R, Z, cstride=1, rstride=1, cmap='viridis')
   ax.view_init(elev=30, azim=frame_number*4)
   #return fig,
   
#%%
ani = animation.FuncAnimation(fig, 
                              change_plot, 
                              90, 
                              fargs=(plot),
                              interval=1)

#%%
ani.save('psf.gif')