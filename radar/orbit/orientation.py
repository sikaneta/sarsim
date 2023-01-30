# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:10:22 2022

@author: Ishuwa.Sikaneta

"""
from space.planets import earth, venus
import numpy as np
import quaternion

# Define the orbit class
class orbit:
    """
    Class to compute the zero Doppler for a given orbit.
    
    The methods of this class implement various theoretical calculations
    found in both `Reference Systems`_ and the Annex of 
    `Pointing Justification`_. 
    
    The class has been designed to be work for any planet, not just Earth.
    
    As derived in `Pointing Justification`_, the computations calculate the
    required look direction of the boresight of the antenna beam that
    satisfies the zero-Doppler requirement and the given off-nadir pointing
    requirement as a function of the orbit angle of the satellite orbit. This
    required look direction is the the required look direction in the inertial
    reference frame, which is the naturaly reference frame in which the
    satellite operates.
    
    Because computations are based on Kepler orbit elements, and the steering
    law has been derived according to an ideal orbit, it may be noted that 
    the derived steering law is only accurate for a real orbit around the 
    initial point at which the object is defined. For instance, if the object
    is initialized with a supplied state vector (provided by AOCS, for 
    instance), then the law will only be valid for a few minutes around this
    point. This is simply because of the inaccuracy of propagating an orbit
    using Kepler orbital elements.

    
    Methods
    -------
    setFromStateVector
        Set the orbit parameters for this object using a state vector. This
        will compute the Kepler orbit elements from a given inertial state 
        vector.
    state2kepler
        Compute the Kepler orbit elements given an inertial state vector.
    computeSV
        Compute the state vector at a given orbit angle
    computeTCN
        Compute the T,C,N vectors at a given orbit angle
    computeV
        Compute the satellite speed at a given orbit angle
    computeAEU
        Compute the look direction, azimuth and elevation vectors at a given 
        orbit angle
    computeE
        Compute the vectors e1 and e2 in B.2.1 of `Pointing Justification`_ at
        the given orbit angle
    computeItoR
        Compute the matrix to transform PCI to PCR reference system at the
        given orbit angle.
    computeT
        Compute the time after the ascending node that corresponds to the 
        given orbit angle
    computeO
        Compute the orbit angle that corresponds to the given time after the
        ascending node
        
    """
    
    angleFunction = {"degrees": np.radians,
                     "radians": lambda x:x}
    
    def __init__(self,
                 e=0,
                 arg_perigee=0,
                 a=10000000.0,
                 period = None,
                 inclination=np.pi/2,
                 planet = earth(),
                 angleUnits = "radians"
                 ):
        """
        Initiator for orbit object. 
        
        It is assumed that the orbit is around the given planet and that
        the orbit is elliptical.

        Parameters
        ----------
        e : float (Unitless)
            The orbit eccentricity.
        arg_perigee : float (angleUnits according to angleUnits)
            Argument of Perigee. This is the angle measured from the vector
            from planet center to the ascending node to the vector from planet
            center to perigee.
        a : float (m)
            The length of the orbit semi-major axis. If this is given but the
            period is not given, then the period will be computed according
            to Kepler's 3rd law.
        period : float (s)
            The period of the orbit in seconds. If this is set, then the value
            for a will be accordingly (Kepler's 3rd law) recomputed.
        inclination : (angleUnits according to angleUnits)
            The inclination angle of the orbit. This is the right-handed angle 
            around the vector from the planet center to the ascending node by 
            which the orbit plane is rotated.
        planet : `radar.space.planet`
            Object that defines the planet. Default is Earth
        angleUnits : `str`
            Units for input angles, degrees or radians.

        Returns
        -------
        None.

        """
        
        self.toRadians = self.angleFunction[angleUnits]
        self.angleUnits = angleUnits
        self.e = e
        self.arg_perigee = self.toRadians(arg_perigee)
        self.inclination = self.toRadians(inclination)
        self.planet = planet
        
        if a is not None:
            self.a = a
            self.period = 2*np.pi*np.sqrt(a**3/planet.GM)
        if period is not None:
            self.period = period
            self.a = (period*np.sqrt(planet.GM)/(2*np.pi))**(2/3)
        
        cosI = np.cos(self.inclination)
        sinI = np.sin(self.inclination)
        cosP = np.cos(self.arg_perigee)
        sinP = np.sin(self.arg_perigee)
        self.rotIfromAnode = np.eye(3)
        self.rotOfromE = np.array([[sinP, -cosP, 0],
                                   [cosP,  sinP, 0],
                                   [0,     0   , 1]])
        self.rotIfromO = np.array([[cosI,  0, sinI],
                                   [0,     1, 0   ],
                                   [-sinI, 0, cosI]])

    
    def setFromStateVector(self, X):
        """
        Set the orbit parameters for this object using a state vector
        
        This function modifies the orbit parameters for this object. The
        Kepler orbit elements are calculated using the state2kepler function

        Parameters
        ----------
        X : `numpy.ndarray` (6,)
            The supplied state vector.

        Returns
        -------
        None.

        """
        
        kepler = self.state2kepler(X)
        
        """ Set everything """
        self.e = kepler["eccentricity"]
        self.arg_perigee = self.toRadians(kepler["perigee"])
        self.a = kepler["a"]
        self.inclination = self.toRadians(kepler["inclination"])
        self.period = 2*np.pi*np.sqrt(self.a**3/self.planet.GM)
        ascendingNode = self.toRadians(kepler["ascendingNode"])
        orbitAngle = (kepler["trueAnomaly"] 
                      + kepler["perigee"])
        
        cosA = np.cos(ascendingNode)
        sinA = np.sin(ascendingNode)
        cosI = np.cos(self.inclination)
        sinI = np.sin(self.inclination)
        cosP = np.cos(self.arg_perigee)
        sinP = np.sin(self.arg_perigee)
        
        self.rotIfromAnode = np.array([[sinA,  cosA, 0],
                                       [-cosA, sinA, 0],
                                       [0,     0   , 1]])
        self.rotOfromE = np.array([[sinP, -cosP, 0],
                                   [cosP,  sinP, 0],
                                   [0,     0   , 1]])
        self.rotIfromO = np.array([[cosI,  0, sinI],
                                   [0,     1, 0   ],
                                   [-sinI, 0, cosI]])
        
        return orbitAngle, ascendingNode
        
        
    def state2kepler(self, X):
        """
        Estimate the Keplerian elements from a state vector

        Parameters
        ----------
        X : `numpy.ndarray` (6,)
            The state vector in an inertial reference frame.

        Returns
        -------
        dict
            has fields: (angleUnits according to self.angleUnits)
                - eccentricity (unitless)
                - a (m)
                - perigee (angleUnits according to self.angleUnits)
                - inclination (angleUnits according to self.angleUnits)
                - ascendingNode: longitude of the ascending node
                  (angleUnits according to self.angleUnits)
                
        Note
        ----
        Calculations are performed according to the following document by 
        `Rene Schwartz <https://downloads.rene-schwarz.com/download/M002-Cartesian_State_Vectors_to_Keplerian_Orbit_Elements.pdf>`_

        """
        
        rvec = X[:3]
        vvec = X[3:]
        vsqr = np.linalg.norm(vvec)**2
        r = np.linalg.norm(rvec)
        
        """ Compute the semi major axis, eq (8) """
        a = 1/(2/r - vsqr/self.planet.GM)
        
        """ Compute the orbital momentum vector, eq (1) """
        hvec = np.cross(rvec, vvec)
        
        """ Compute the eccentricity vector and eccentricity, eq (2)"""
        evec = np.cross(vvec, hvec)/self.planet.GM - rvec/r
        e = np.linalg.norm(evec)
        
        """ Compute the n vector, eq (3a) """
        nvec = np.cross(np.array([0,0,1]), hvec)
        unvec = nvec/np.linalg.norm(nvec)
        
        """ Compute the true anomaly, eq (3b) """
        urvec = rvec/r
        uevec = evec/np.linalg.norm(evec)
        tAnomaly = np.arccos(uevec.dot(urvec))
        if rvec.dot(vvec) < 0:
            tAnomaly = 2*np.pi-tAnomaly
            
        """ Compute the orbit inclination angle, eq (4) """
        i = np.arccos(hvec[-1]/np.linalg.norm(hvec))
        
        """ Compute the longitude of the ascending node, eq (6a) """
        anode = np.arccos(unvec[0])
        """ Have added the pi above because I was getting
            the descending node instead. Check for reason. """
        anode = anode if nvec[1]>=0 else 2*np.pi - anode
        
        """ Compute the angle of periapsis, eq (6b) """
        p = np.arccos(unvec.dot(uevec))
        p = p if evec[-1] >= 0 else 2*np.pi - p
        
        angleUnitFn = (lambda x:x) if self.toRadians(1) == 1 else np.degrees
        return {"eccentricity": e,
                "perigee": angleUnitFn(p),
                "a": a,
                "inclination": angleUnitFn(i),
                "ascendingNode": angleUnitFn(anode),
                "trueAnomaly": angleUnitFn(tAnomaly)}
    
    def computeSV(self, beta):
        """
        Compute the state vector at the orbit angle
        
        Compute the satellite state vector at the given orbit angle. This
        vector is given in the Inertial coordinate system

        Parameters
        ----------
        beta : float, (angleUnits according to self.angleUnits)
            The orbit angle measure from the ascending node.

        Returns
        -------
        state: float
            The state vector at the orbit angle
        r: float
            The range from the center of the planet to the satellite in (m)
        v: float
            The speed of the satellite in the Inertial reference frame and
            at the supplied orbit angle

        """
        
        a = self.a
        e = self.e
        GM = self.planet.GM
        U = self.toRadians(beta) - self.arg_perigee
        cosB = np.cos(self.toRadians(beta))
        sinB = np.sin(self.toRadians(beta))
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
        
        """ Rotate by the longitude of the ascending node
            to get into inertial coordinate system """
        mInertial = self.rotIfromAnode.dot(self.rotIfromO)
        
        state = np.hstack((np.dot(mInertial, Xo),
                           np.dot(mInertial, Vo)))
        return state, r, v
    
    def computeTCN(self, beta):
        """
        Compute the T,C,N vectors
        
        This function computes the unit vectors that define the T,C,N
        reference system. These are vectors in the PCI reference
        sytem

        Parameters
        ----------
        beta : float, (angleUnits according to self.angleUnits)
            The orbit angle at which to compute the T,C,N vectors. This is the
            angle measured from the ascending node.

        Returns
        -------
        `np.ndarray` [3,3] 
            A numpy array with the T,C,N vectors as columns in the respective
            order.

        """
        
        X, _, _ = self.computeSV(beta)
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
        beta : float, (angleUnits according to self.angleUnits)
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
        return np.sqrt((GM/a)*(1+2*e*np.cos(self.toRadians(beta)-w) 
                               + e**2)/(1-e**2))
    
    def computeAEU(self, beta, off_nadir):
        """
        Compute the look direction, azimuth and elevation vectors
        
        Compute the look direction, azimuth and elevation vectors in
        both the TCN frame and the Inertial frame

        Parameters
        ----------
        beta : float (angleUnits according to self.angleUnits)
            The orbit angle. Measured from the ascending node.
        off_nadir : float (angleUnits according to self.angleUnits)
            Off-nadir angle. This is signed-angle corresponding to a roll.
            A negative angle corresponds to right-looking, a positive
            angle corresponds to left-looking. This is the right-handed rule
            for angles rotated around the velocity vector under the assumption
            that the default (0) is looking nadir.
            
        Returns
        -------
        aeuI: `npmpy.array`, [3,3]
            The azimuth, elevation and look vectors in the Inertial reference
            frame arranged as the columns of a matrix (respective order)
        aeuTCN: `npmpy.array`, [3,3]
            The azimuth, elevation and look vectors in the TCN frame arranged 
            as the columns of a matrix (respective order)
                
        Notes
        -----
        The calculation coded here multiples the equations in the Annex of
        `Pointing Justification`_ by the planet angular velocity to allow the 
        computations even in the case the planet angular velocity is low or 
        zero.

        """
        
        look = -np.sign(off_nadir)
        v = np.cos(self.toRadians(off_nadir))
        U = self.toRadians(beta) - self.arg_perigee
        e = self.e
        w = self.planet.w
        wC = np.sqrt(self.planet.GM/(self.a*(1-e**2))**3)
        cosU = np.cos(U)
        sinU = np.sin(U)
        cosI = np.cos(self.inclination)
        sinI = np.sin(self.inclination)
        cosB = np.cos(self.toRadians(beta))
        
        """ Compute equations (107) and (108). Eq numbers may change! """
        cosG = (1+e*cosU)/np.sqrt(1+2*e*cosU+e**2)
        sinG = e*sinU/np.sqrt(1+2*e*cosU+e**2)
        
        """ Compute equations (117) and (118). Eq numbers may change!  """
        wDcosG = wC*(1+e*cosU)**2
        wDsinG = e*wC*(1+e*cosU)*sinU
        wD = np.sqrt(wDcosG**2 + wDsinG**2)
        
        """ Compute equation (75) """
        wF = wDcosG - w*cosI
        
        """ Compute equation (79). s2, s2F can be plus/minus depending on look
            direction. wwQ is the denominator of (78,79) multiplied by
            w^2 """
        wwQ = (w*sinI*cosB)**2 + wF**2
        s2 = look*w*np.sqrt((1-v**2)*wwQ - (w*v*sinG)**2)/wwQ
        s2F = look*wF*np.sqrt((1-v**2)*wwQ - (w*v*sinG)**2)/wwQ
        
        """ Compute the overall result (81)"""
        uTCN = np.array([sinG*v*(wDcosG*wF/wwQ - 1) + s2*cosG*cosB*sinI,
                          -w*v*wDsinG*sinI*cosB/wwQ + s2F,
                          v*(cosG + wDsinG*sinG*wF/wwQ) + s2*sinG*cosB*sinI])
        uTCN /= np.linalg.norm(uTCN)
        
        """ Compute the TCN vectors """
        TCN = self.computeTCN(beta)
        
        
        aTCN = np.array([wD - w*cosG*cosI,
                          -w*cosB*sinI,
                          -w*sinG*cosI])
        aTCN = aTCN/np.linalg.norm(aTCN)
        
        """ Compute the eTCN vector """
        eTCN = np.cross(uTCN, aTCN)
        eTCN/= np.linalg.norm(eTCN)
        
        """ Compute the aeuTCN matrix """
        aeuTCN = np.stack((aTCN, eTCN, uTCN), axis=1)
        
        """ compute the aeuI matrix """
        aeuI = TCN.dot(aeuTCN)
        
        """ Return the aeuI and aeuTCN matrices """
        return aeuI, aeuTCN
    
    def computeE(self, beta, v):
        """
        Compute the e1 and e2
        
        Compute the values of the e1 and e2 unit vectors from  of 
        `Pointing Justification`_

        Parameters
        ----------
        beta : float (angleUnits according to self.angleUnits)
            The orbit angle. Measured from the ascending node.
        v : float (Unitless)
            Cosine of the off-nadir angle
            
        Returns
        -------
        s: float
            The scaling factor
            
        Notes
        -----
        The calculation coded here multiplies the equations in the Annex of
        `Pointing Justification`_ by the planet angular velocity to allow the 
        computations even in the case the planet angular velocity is low or 
        zero.

        """
        
        U = self.toRadians(beta) - self.arg_perigee
        e = self.e
        w = self.planet.w
        wC = np.sqrt(self.planet.GM/(self.a*(1-e))**3)
        cosU = np.cos(U)
        sinU = np.sin(U)
        cosI = np.cos(self.inclination)
        sinI = np.sin(self.inclination)
        cosB = np.cos(self.toRadians(beta))
        
        """ Compute equations (73) and (74). Equation numbers may change! """
        cosG = (1+e*cosU)/np.sqrt(1+2*e*cosU+e**2)
        sinG = e*sinU/np.sqrt(1+2*e*cosU+e**2)
        
        """ Compute equations (79) and (80). Equation numbers may change!  """
        wDcosG = wC*(1+e*cosU)**2
        wDsinG = e*wC*(1+e*cosU)*sinU
        wD = np.sqrt(wDcosG**2 + wDsinG**2)
        
        e1 = np.array([wD-w*cosI*cosG, -w*sinI*cosB, -w*cosI*sinG])
        e2 = np.array([-sinG,
                       0,
                       cosG])
        
        return e1, e2 
    
    def computeItoR(self, beta):
        """
        Compute the matrix to transform PCI to PCR reference system
        
        This function returns the rotation matrix and its time derivative
        to go from the Planet Centered Inertial coordinate system to the 
        Planet Centered Rotating coordinate system.

        Parameters
        ----------
        beta : float (angleUnits according to self.angleUnits)
            The orbit angle (measured from the ascending node) for which to
            compute the transformation matrix.

        Returns
        -------
        M : `np.ndarray` [3,3]
            The rotation matrix.
        dM : `np.ndarray` [3,3]
            The time derivative of the rotation matrix.

        """
        
        t = self.computeT(beta)
        cosWT = np.cos(self.planet.w*t)
        sinWT = np.sin(self.planet.w*t)
        M = np.array([[cosWT,  sinWT, 0],
                      [-sinWT, cosWT, 0],
                      [0,          0, 1]])
        
        dM = self.planet.w*np.array([[-sinWT, cosWT,  0],
                                     [-cosWT, -sinWT, 0],
                                     [0,          0,  0]])
        return M, dM
    
    def computeT(self, beta):
        """
        Compute time given the orbit angle.
        
        Computes the time from perigee given the orbit angle,
        beta, measured from the ascending node.

        Parameters
        ----------
        beta : float (angleUnits according to self.angleUnits)
            Orbit angle measured from the ascending node.

        Returns
        -------
        float
            The amount of time passed to get to the orbit angle.

        """
        
        U = self.toRadians(beta) - self.arg_perigee
        
        a = self.a
        e = self.e
        g = 1-e**2
        W = np.sqrt(self.planet.GM/(a**3*g))
        # t1 = 2/np.sqrt(g)*np.arctan((1-e)*np.tan(U/2)/np.sqrt(g)) 
        t1 = 2/np.sqrt(g)*np.arctan2((1-e)*np.tan(U/2), np.sqrt(g))
        t2 = e*np.sin(U)/(1+e*np.cos(U))
        return 1/(W)*(t1-t2)
    
    def computeO(self, t, error = 1e-12, max_iter = 10):
        """
        Compute the orbit angle from time
        
        Compute the orbit angle from time using the Newton-Raphson
        iterative method

        Parameters
        ----------
        t : float
            The time in seconds since the time of the ascending node at
            which to compute the orbit angle.

        Returns
        -------
        The orbit angle (angleUnits according to self.angleUnits).

        """
        
        """ Normalize to the length of the period """
        t -= np.round(t/self.period)*self.period
        a = self.a
        e = self.e
        g = 1-e**2
        W = np.sqrt(self.planet.GM/(a**3*g))
        
        U_old = np.pi/2
        for k in range(max_iter):
            c1 = (1+e*np.cos(U_old))
            c2 = np.sin(U_old)
            beta = U_old + self.arg_perigee
            beta = beta if self.toRadians(1)==1 else np.degrees(beta)
            tn = self.computeT(beta)
            U_new = U_old + (t-tn)*W*c1**2
            U_new = U_old - c1/(2*e*c2)*(1-np.sqrt(1+4*e*W*c1*c2*(t-tn)))
            iter_error = np.abs(U_new-U_old)
            if iter_error < error:
                break
            U_old = U_new
            
        beta = np.mod(beta, 2*np.pi if self.toRadians(1)==1 else 360)
        return beta, iter_error

