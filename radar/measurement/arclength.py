# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:36:20 2019

@author: SIKANETAI
"""
import numpy as np

#%% Define a class for the slow time arclength
class slow:
    def __init__(self, datetimes=[]):
        self.N = len(datetimes)
        self.refIDX = int(self.N/2)
        self.t = np.array([(t - datetimes[self.refIDX])/np.timedelta64(1,'s') 
                           for t in datetimes])
    
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
        
    def s2t(self, s, max_iter=10, errortol = 1e-9):
        tdf = self.tdf
        def f(t):
            return tdf[0] + t*tdf[1] + 0.5*t**2*tdf[2] + 1.0/6.0*t**3*tdf[3] - s
        def df(t):
            return tdf[1] + t*tdf[2] + 0.5*t**2*tdf[3]
        def ddf(t):
            return tdf[2] + t*tdf[3]
        
        """ Start with an initial guess """
        t0 = s/tdf[1]
        
        for k in range(max_iter):
            t = t0 - (df(t0) - np.sqrt(df(t0)**2 - 2*f(t0)*ddf(t0)))/ddf(t0)
            error = np.max(np.abs(t - t0))
            if error<errortol:
                break
            else:
                t0 = t
        
        if k == max_iter:
            print("Warning!. arclength to time failed to converge!")
        
        return t
        
    def sFromX(self, X, squint = 0.0):
        print("Squint")
        print(squint)
        cdf = self.cdf
        def c(s):
            return cdf[0] + cdf[1]*s + cdf[2]*s**2/2.0 + cdf[3]*s**3/6.0
        
        def cdot(s):
            return cdf[1] + cdf[2]*s + cdf[3]*s**2/2.0
        
        def cddot(s):
            return cdf[2] + cdf[3]*s
        
        def cdddot(s):
            return cdf[3]

        def f(s, X):
            return np.dot(cdot(s), c(s)-X) - squint
        
        def fdot(s,X):
            return np.dot(cddot(s), c(s)-X) + np.dot(cdot(s),cdot(s))
        
        def fddot(s,X):
            return np.dot(cdddot(s), c(s)-X) + 3*np.dot(cddot(s),cdot(s))
        
        def iterateS(s, X):
            fn = f(s, X)
            fdn = fdot(s,X)
            fddn = fddot(s,X)
            return s + (-fdn + np.sqrt(fdn**2 - 2*fn*fddn))/fddn
        
        def newtonS(X, s=0.0, errorTol=1e-12, maxIter = 10):
            sn = s
            for k in range(maxIter):
                s = iterateS(sn, X)
                error = np.abs(s - sn)
                if error < errorTol:
                    break
                else:
                    sn = s
            if k == maxIter:
                print("Warning: Newton algorithm failed to converge")
            return s
        
        m,n = X.shape
        S = np.zeros(X.shape)
        R = np.zeros(X.shape)
        s = np.zeros(n)
        for k in range(n):
            s[k] = newtonS(X[:,k])
            S[:,k] = c(s[k])
            R[:,k] = S[:,k] - X[:,k]
        return R, S, s