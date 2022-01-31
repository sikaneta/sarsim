# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:10:22 2022

@author: Ishuwa.Sikaneta
"""

from space.planets import earth
import numpy as np

class orbit:
    """
    Class to compute the zero Doppler for a given orbit
    """
    
    def __init__(self,
                 e,
                 arg_perigee,
                 a,
                 inclination,
                 planet = earth
                 ):
        """
        Initiator for orbit object. 
        
        It is assumed that the orbit is around the given planet and that
        the orbit is elliptical.

        Parameters
        ----------
        e : float (Unitless)
            The orbit eccentricity.
        arg_perigee : float (Radians)
            Argument of Perigee. This is the angle measured from the vector
            from planet center to the ascending node to the vector from planet
            center to perigee.
        a : float (m)
            The length of the orbit semi-major axis.
        inclination : (Radians)
            The inclination angle of the orbit. This is the right-handed angle 
            around the vector from the planet center to the ascending node by 
            which the orbit plane is rotated.
        planet : `radar.space.planet`
            Object that defines the planet. Default is Earth

        Returns
        -------
        None.

        """
        self.e = e
        self.arg_perigee = arg_perigee
        self.a = a
        self.inclination = inclination
        self.planet = planet
        self.period = 2*np.pi*np.sqrt(a**3/planet.GM)
        cosI = np.cos(inclination)
        sinI = np.sin(inclination)
        cosP = np.cos(arg_perigee)
        sinP = np.sin(arg_perigee)
        self.rotOfromE = np.array([[sinP, -cosP, 0],
                                   [cosP,  sinP, 0],
                                   [0,     0   , 1]])
        self.rotIfromO = np.array([[cosI,  0, sinI],
                                   [0,     1, 0   ],
                                   [-sinI, 0, cosI]])
    
    def fromPeriod(self, period):
        self.period = period
        self.a = ((period/2/np.pi)**2*self.planet.GM)**(1/3)
    
    def computeR(self, beta):
        """
        Compute the range at the orbit angle
        
        Compute the satellite range at the specified orbit angle beta

        Parameters
        ----------
        beta : float, (Radians)
            The orbit angle measure from the ascending node.

        Returns
        -------
        float:
            The range from the center of the planet to the satellite in (m)

        """
        a = self.a
        e = self.e
        GM = self.planet.GM
        U = beta - self.arg_perigee
        cosB = np.cos(beta)
        sinB = np.sin(beta)
        cosP = np.cos(self.arg_perigee)
        sinP = np.sin(self.arg_perigee)
        
        """ Compute amplitude of position and velocity """
        v = self.computeV(beta)
        r = self.a*(1-self.e**2)/(1+self.e*np.cos(U)) 
        
        """ Compute the position and velocity vectors in OCS frame """
        Xo = r*np.array([-sinB, cosB, 0])
        Vo = -np.sqrt(GM/a/(1-e**2))*np.array([cosB + e*cosP,
                                               sinB + e*sinP,
                                               0])
        
        state = np.hstack((np.dot(self.rotIfromO, Xo),
                           np.dot(self.rotIfromO, Vo)))
        return state, r, v
    
    def computeTCN(self, beta):
        X, _, _ = self.computeR(beta)
        Xhat = X[0:3]/np.linalg.norm(X[0:3])
        T = X[3:]/np.linalg.norm(X[3:])
        C = np.cross(T, Xhat)
        C = C/np.linalg.norm(C)
        N = np.cross(T,C)
        N = N/np.linalg.norm(N)
        return np.stack((T,C,N), axis=1)
    
    def computeV(self, beta):
        """
        Compute the satellite speed at the given orbit angle

        Parameters
        ----------
        beta : float, (Radians)
            The orbit angle measure from the ascending node.

        Returns
        -------
        float:
            The orbit speed in (m/s)

        """
        a = self.a
        e = self.e
        w = self.arg_perigee
        GM = self.planet.GM
        return np.sqrt((GM/a)*(1+2*e*np.cos(beta-w) + e**2)/(1-e**2))
        
    def computeU(self, beta, v):
        """
        Compute the scaling parameter for solving the underdetermined system
        of equations
        
        Compute the value of s in equation (56)

        Parameters
        ----------
        beta : float (Radias)
            The orbit angle. Measured from the ascending node.
        v : float (Unitless)
            Cosine of the off-nadir angle
        Returns
        -------
        s: float
            The scaling factor

        """
        
        U = beta - self.arg_perigee
        e = self.e
        C = 1.0/self.planet.w*np.sqrt(self.planet.GM/(self.a*(1-e))**3)
        cosU = np.cos(U)
        sinU = np.sin(U)
        cosI = np.cos(self.inclination)
        sinI = np.sin(self.inclination)
        cosB = np.cos(beta)
        
        """ Compute equations (73) and (74). Equation numbers may change! """
        cosG = (1+e*cosU)/np.sqrt(1+2*e*cosU+e**2)
        sinG = e*sinU/np.sqrt(1+2*e*cosU+e**2)
        
        """ Compute equations (79) and (80). Equation numbers may change!  """
        DcosG = C*(1+e*cosU)**2
        DsinG = e*C*(1+e*cosU)*sinU
        
        """ Compute equation (56) """
        A = DcosG - cosI
        Q = (sinI*cosB)**2 + A**2
        
        """ Compute equation (59) """
        sB = np.sqrt((1-v**2)*Q - (v*DsinG)**2)/Q
        
        """ Compute the overall result """
        u = np.array([sinG*v*(DcosG*A/Q - 1) + sB*cosG*cosB*sinI,
                      -v*DsinG*sinI*cosB/Q + sB*A,
                      v*(cosG + DsinG*sinG*A/Q) + sB*sinG*cosB*sinI])
        
        return self.computeTCN(beta).dot(u), u
    
    def computeE(self, beta, v):
        """
        Compute the scaling parameter for solving the underdetermined system
        of equations
        
        Compute the value of s in equation (56)

        Parameters
        ----------
        beta : float (Radias)
            The orbit angle. Measured from the ascending node.
        v : float (Unitless)
            Cosine of the off-nadir angle
        Returns
        -------
        s: float
            The scaling factor

        """
        
        U = beta - self.arg_perigee
        e = self.e
        C = 1.0/self.planet.w*np.sqrt(self.planet.GM/(self.a*(1-e))**3)
        cosU = np.cos(U)
        sinU = np.sin(U)
        cosI = np.cos(self.inclination)
        sinI = np.sin(self.inclination)
        cosB = np.cos(beta)
        
        """ Compute equations (73) and (74). Equation numbers may change! """
        cosG = (1+e*cosU)/np.sqrt(1+2*e*cosU+e**2)
        sinG = e*sinU/np.sqrt(1+2*e*cosU+e**2)
        
        """ Compute equations (79) and (80). Equation numbers may change!  """
        DcosG = C*(1+e*cosU)**2
        DsinG = e*C*(1+e*cosU)*sinU
        D = np.sqrt(DcosG**2 + DsinG**2)
        
        e1 = np.array([D-cosI*cosG, -sinI*cosB, -cosI*sinG])
        e2 = np.array([-sinG,
                       0,
                       cosG])
        
        return e1, e2
        

    def computeT(self, beta):
        """
        Compute time given the orbit abgle
        
        Computes the time from the ascending node given the orbit angle,
        also measured from the ascending node.

        Parameters
        ----------
        beta : float (rad)
            Orbit angle measured from the ascending node.

        Returns
        -------
        float
            The amount of time passed to get to the orbit angle.

        """
        a = self.a
        e = self.e
        g = 1-e**2
        W = np.sqrt(self.planet.GM/(a**3*g))
        t1 = 2/np.sqrt(g)*np.arctan((1-e)*np.tan(beta/2)/np.sqrt(g)) 
        t2 = e*np.sin(beta)/(1+e*np.cos(beta))
        return 1/(W)*(t1-t2)
        
        
#%% Tests   
""" Some tests """
off_nadir = 30
orbit_angle = 79.7417+80

sentinel = orbit(0.0001303, 
                 np.radians(79.7417), 
                 7167100, 
                 np.radians(98.1813))
X, r, v = sentinel.computeR(np.radians(orbit_angle))

print("|r|: %0.6f, r: %0.6f" % (np.linalg.norm(X[0:3]), r))
print("|v|: %0.6f, v: %0.6f" % (np.linalg.norm(X[3:]), v))

uI, u = sentinel.computeU(np.radians(orbit_angle), np.cos(np.radians(off_nadir)))
e1, e2 = sentinel.computeE(np.radians(orbit_angle), np.cos(np.radians(off_nadir)))
print("Norm of u: %0.6f" % np.linalg.norm(u))
print("u*e1: %0.6f" % np.dot(u,e1))
print("u*e2: %0.6f, v: %0.6f" % (np.dot(u,e2), np.cos(np.radians(off_nadir))))

#%%
beta = np.radians(np.arange(0,720,0.00001))
t = sentinel.computeT(beta)
print(max(t)-min(t))
print(2*np.pi*np.sqrt(sentinel.a**3/sentinel.planet.GM))

