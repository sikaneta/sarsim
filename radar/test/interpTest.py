# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 08:10:38 2019

@author: SIKANETAI
"""

import configuration.configuration as cfg
import numpy as np
import time
import matplotlib.pyplot as plt

#%%
newxml = "e:\\Users\\SIKANETAI\\simulation\\30cm\\reduced_30cm_simulation.xml"
radar = cfg.loadConfiguration(newxml)
bands = np.arange(-int(len(radar)/2)-2,int(len(radar)/2)+1+2)
r_sys = cfg.radar_system(radar, bands)
r_sys.computeGroundPoint(radar, range_idx=400)

#%% local stuff
[KR,KX] = np.meshgrid(r_sys.kr_sorted, r_sys.ks_full)

#%% Get the initial guess
KR_n = np.sqrt(KR**2+KX**2/a2)
X_n = r/np.sqrt(a2)*KX/np.sqrt(a2*KR**2-KX**2)

#%% Define some constants
bt_sqr = r**2/a2
gamma = -np.sqrt(a2)*a3/2.0
xi = a2*a4

#%% Define the g function
def g(s):
    return np.sqrt(a2 + a3*s + a4*s**2)

#def of(x, C):
#    return x/np.sqrt(x**2+r**2/a2)*(g(-x) - x*np.sqrt(a2)*a3/2.0/g(-x)**2 + a2*a4*x**2/g(-x)**3) - C


#%% Define the kernel and derivatives
def kernel(x,k,l,m):
    return x**k/(r**2+a2*x**2)**l/g(-x)**m

def dkernel(x,k,l,m):
    return (k*kernel(x,k-1,l,m) 
            - 2.0*a2*l*kernel(x,k+1,l+1,m) 
            + a3*m/2.0*kernel(x,k,l,m+2)
            -a4*m*kernel(x,k+1,l,m+2))
    
#def ddkernel(x,k,l,m):
#    if k==0:
#        return (-2.0*a2*l*dkernel(x,k+1,l+1,m) 
#            + a3*m/2.0*dkernel(x,k,l,m+2)
#            -a4*m*dkernel(x,k+1,l,m+2))
#    return (k*dkernel(x,k-1,l,m) 
#            - 2.0*a2*l*dkernel(x,k+1,l+1,m) 
#            + a3*m/2.0*dkernel(x,k,l,m+2)
#            -a4*m*dkernel(x,k+1,l,m+2))
    
def dkernel_factored(x,k,l,m):
    return kernel(x,k-1,l,m)*(k - 2*a2*l*x**2/(r**2+a2*x**2) + m*(a3/2.0-a4*x)/g(-x)**2)
    

#%% Define the function to be inverted for x
def f(x, C):
    return kernel(x,1,0.5,-1) + gamma*kernel(x,2,0.5,2) + xi*kernel(x,3,0.5,3) - C/np.sqrt(a2)

def df(x):
    return dkernel_factored(x,1,0.5,-1) + gamma*dkernel_factored(x,2,0.5,2) + xi*dkernel_factored(x,3,0.5,3)
#def df(x):
#    return dkernel(x,1,0.5,-1) + gamma*dkernel(x,2,0.5,2) + xi*dkernel(x,3,0.5,3)

def ddf(x):
    return ddkernel(x,1,0.5,-1) + gamma*ddkernel(x,2,0.5,2) + xi*ddkernel(x,3,0.5,3)

#%% Iterate test
my_X = XX[100000,100]

#%% Newton method
t = time.time()
print("Calculating the function at current X_n")
f_xn = f(my_X, KX[100000,100]/KR[100000,100])

print("Calculating the derivative of the function at current X_n")
df_xn = df(my_X)

my_X = my_X - f_xn/df_xn

print("Print %0.15f" % my_X)
print("Iteration time")
print(time.time()-t)

#%% Newton method
t = time.time()
print("Calculating the function at current X_n")
f_xn = f(XX, KX/KR)

print("Calculating the derivative of the function at current X_n")
df_xn = df(XX)

XX = XX - f_xn/df_xn

print("Max error: %0.9f" % np.max(np.abs(f_xn/df_xn)))

print("Iteration time")
print(time.time()-t)

#%%
KR_int = KR*np.sqrt(1+a2/r**2*XX**2) - KX*np.sqrt(a2)/r*XX/g(-XX)

#%%
idx0 = 600
idx1 = 610
plt.figure()
plt.plot(KR_new[:,idx0:idx1] - KR_n[:,idx0:idx1])
plt.grid()
plt.show()