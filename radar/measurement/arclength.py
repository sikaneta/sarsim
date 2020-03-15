# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:36:20 2019

@author: SIKANETAI
"""
import datetime
import numpy as np

#%% Define a class for the slow time arclength
class slow:
    def __init__(self, datetimes):
        self.N = len(datetimes)
        self.refIDX = int(self.N/2)
        self.ref = datetimes[self.refIDX]
        self.t = np.array([(t - self.ref)/np.timedelta64(1,'s') for t in datetimes])
    
    def diffG(self, exp_state):
        X = exp_state[0,:]
        dX = exp_state[1,:]
        ddX = exp_state[2,:]
        dddX = exp_state[3,:]
        norm_v = np.linalg.norm(dX)
        v = dX/norm_v

        Pv = np.eye(3) - np.outer(dX,dX)/norm_v**2
        kappa = np.linalg.norm(np.dot(Pv,ddX))/norm_v**2

        T = dX/norm_v
        N = np.dot(Pv, ddX)
        N = N/np.linalg.norm(N)
        B = np.cross(T,N)

        w = np.dot(Pv, ddX)
        norm_w = np.linalg.norm(w)
        Pw = np.eye(3) - np.outer(w,w)/norm_w**2
        dmy = np.dot(Pv, np.outer(ddX, v))
        dw = -1.0/norm_v*np.dot(dmy + dmy.T, ddX) + np.dot(Pv, dddX)
        dN = np.dot(Pw, dw)/norm_v/norm_w
        dkappa = np.dot(dw,N)/norm_v**3 - 2*norm_w*np.dot(ddX,T)/norm_v**5
        tau = np.dot(dN,B)
        print("kappa: %0.9e tau: %0.9e dkappa: %0.9e" % (kappa, tau, dkappa))
        cdf = [X, T, kappa*N, -kappa**2*T + dkappa*N + kappa*tau*B]
        tdf = [0.0, norm_v, np.dot(ddX,v), np.dot(ddX,w)/norm_v + np.dot(dddX,v)]
        
        #set the values for this object
        self.cdf = cdf
        self.tdf = tdf 
        self.T = T 
        self.N = N 
        self.B = B 
        self.kappa = kappa 
        self.tau = tau 
        self.dkappa = dkappa
        return cdf, tdf, T, N, B, kappa, tau, dkappa
    
    def t2s(self):
        t = self.t
        tdf = self.tdf
        cdf = self.cdf
        s = tdf[0] + t*tdf[1] + 0.5*t**2*tdf[2] + 1.0/6.0*t**3*tdf[3]
        c = np.outer(cdf[0],s**0) + np.outer(cdf[1],s**1) + 0.5*np.outer(cdf[2],s**2) + 1.0/6.0*np.outer(cdf[3], s**3)
        self.s = s
        self.c = c
        return
    
    def ds(self, t):
        tdf = self.tdf
        return tdf[0] + t*tdf[1] + 0.5*t**2*tdf[2] + 1.0/6.0*t**3*tdf[3]
    
    def computeRangeCoefficients(self, R):
        self.a2 = 1.0-self.kappa*R.dot(self.N)
        self.a3 = -(R.dot(self.B)*self.kappa*self.tau+R.dot(self.N)*self.dkappa)/3.0
        self.a4 = -self.kappa**2/12.0
        self.R = -R