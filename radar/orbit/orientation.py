# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:10:22 2022

@author: Ishuwa.Sikaneta
"""

from space.planets import earth, venus
import numpy as np
import quaternion
import matplotlib.pyplot as plt

#%% Define the orbit class
class orbit:
    """
    Class to compute the zero Doppler for a given orbit
    """
    
    angleFunction = {"degrees": np.radians,
                     "radians": lambda x:x}
    
    def __init__(self,
                 e=0,
                 arg_perigee=0,
                 a=10000000.0,
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
        angleUnits : `str`
            Units for input angles, degrees or radians.

        Returns
        -------
        None.

        """
        
        self.toRadians = self.angleFunction[angleUnits]
        self.e = e
        self.arg_perigee = self.toRadians(arg_perigee)
        self.a = a
        self.inclination = self.toRadians(inclination)
        self.planet = planet
        self.period = 2*np.pi*np.sqrt(a**3/planet.GM)
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
            has fields: (radians or degress according to self.toRadians 
            definition)
                - eccentricity (unitless)
                - a (m)
                - perigee (radians or degrees)
                - inclination (radians or degrees)
                - ascendingNode (radians or degrees)
                
        Note
        ----
        Calculations are performed according to 
        `this reference <https://downloads.rene-schwarz.com/download/M002-Cartesian_State_Vectors_to_Keplerian_Orbit_Elements.pdf>`_

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

           
    def fromPeriod(self, period):
        self.period = period
        self.a = ((period/2/np.pi)**2*self.planet.GM)**(1/3)
    
    def computeR(self, beta):
        """
        Compute the state vector at the orbit angle
        
        Compute the satellite state vector at the given orbit angle. This
        vector is given in the Inertial coordinate system

        Parameters
        ----------
        beta : float, (Radians)
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
        beta : float, (Radians)
            The orbit angle at which to compute the T,C,N vectors. This is the
            angle measured from the ascending node.

        Returns
        -------
        `np.ndarray` [3,3] 
            A numpy array with the T,C,N vectors as columns in the respective
            order.

        """
        
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
        return np.sqrt((GM/a)*(1+2*e*np.cos(self.toRadians(beta)-w) 
                               + e**2)/(1-e**2))
              
    
    def computeAEUold(self, beta, v):
        """
        Compute the look direction, azimuth and elevation vectors
        
        Compute the look direction, azimuth and elevation vectors in
        both the TCN frame and the Inertial frame

        Parameters
        ----------
        beta : float (Radians)
            The orbit angle. Measured from the ascending node.
        v : float (Unitless)
            Cosine of the off-nadir angle
        Returns
        -------
        aeuI: `npmpy.array`, [3,3]
            The azimuth, elevation and look vectors in the Inertial reference
            frame arranged as the columns of a matrix (respective order)
        aeuTCN: `npmpy.array`, [3,3]
            The azimuth, elevation and look vectors in the TCN frame arranged 
            as the columns of a matrix (respective order)

        """
        
        U = self.toRadians(beta) - self.arg_perigee
        e = self.e
        C = 1.0/self.planet.w*np.sqrt(self.planet.GM/(self.a*(1-e**2))**3)
        cosU = np.cos(U)
        sinU = np.sin(U)
        cosI = np.cos(self.inclination)
        sinI = np.sin(self.inclination)
        cosB = np.cos(self.toRadians(beta))
        
        """ Compute equations (89) and (90). Equation numbers may change! """
        cosG = (1+e*cosU)/np.sqrt(1+2*e*cosU+e**2)
        sinG = e*sinU/np.sqrt(1+2*e*cosU+e**2)
        
        """ Compute equations (99) and (100). Equation numbers may change! """
        DcosG = C*(1+e*cosU)**2
        DsinG = e*C*(1+e*cosU)*sinU
        D = np.sqrt(DcosG**2 + DsinG**2)
        
        """ Compute equation (54) """
        A = DcosG - cosI
        
        """ Compute equation (55). sB can be plus/minus depending on look
            direction """
        Q = (sinI*cosB)**2 + A**2
        sB = np.sqrt((1-v**2)*Q - (v*sinG)**2)/Q
        
        """ Compute the overall result (60)"""
        uTCN = np.array([sinG*v*(DcosG*A/Q - 1) + sB*cosG*cosB*sinI,
                         -v*DsinG*sinI*cosB/Q + sB*A,
                         v*(cosG + DsinG*sinG*A/Q) + sB*sinG*cosB*sinI])
        
        """ Compute the TCN vectors """
        TCN = self.computeTCN(beta)
        
        
        aTCN = np.array([D - cosG*cosI,
                         -cosB*sinI,
                         -sinG*cosI])
        aTCN = aTCN/np.linalg.norm(aTCN)
        
        """ Compute the eTCN vector """
        eTCN = np.cross(uTCN, aTCN)
        
        """ Compute the aeuTCN matrix """
        aeuTCN = np.stack((aTCN, eTCN, uTCN), axis=1)
        
        """ compute the aeuI matrix """
        aeuI = TCN.dot(aeuTCN)
        
        """ Return the aeuI and aeuTCN matrices """
        return aeuI, aeuTCN
    
    def computeAEU(self, beta, off_nadir):
        """
        Compute the look direction, azimuth and elevation vectors
        
        Compute the look direction, azimuth and elevation vectors in
        both the TCN frame and the Inertial frame

        Parameters
        ----------
        beta : float (Radians)
            The orbit angle. Measured from the ascending node.
        off_nadir : float (Unitless)
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
        The calculation coded here multiples the equations in the notes
        by the planet angular velocity to allow the computation of equation
        (60) even in the case the planet angular velocity is low or zero.

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
    
    # def computeAEU(self, beta, v, look=1):
    #     """
    #     Compute the look direction, azimuth and elevation vectors
        
    #     Compute the look direction, azimuth and elevation vectors in
    #     both the TCN frame and the Inertial frame

    #     Parameters
    #     ----------
    #     beta : float (Radians)
    #         The orbit angle. Measured from the ascending node.
    #     v : float (Unitless)
    #         Cosine of the off-nadir angle
    #     look : int
    #         The look direction vector. +1 for right, -1 for left
            
    #     Returns
    #     -------
    #     aeuI: `npmpy.array`, [3,3]
    #         The azimuth, elevation and look vectors in the Inertial reference
    #         frame arranged as the columns of a matrix (respective order)
    #     aeuTCN: `npmpy.array`, [3,3]
    #         The azimuth, elevation and look vectors in the TCN frame arranged 
    #         as the columns of a matrix (respective order)
                
    #     Notes
    #     -----
    #     The calculation coded here multiples the equations in the notes
    #     by the planet angular velocity to allow the computation of equation
    #     (60) even in the case the planet angular velocity is low or zero.

    #     """
        
    #     U = self.toRadians(beta) - self.arg_perigee
    #     e = self.e
    #     w = self.planet.w
    #     wC = np.sqrt(self.planet.GM/(self.a*(1-e**2))**3)
    #     cosU = np.cos(U)
    #     sinU = np.sin(U)
    #     cosI = np.cos(self.inclination)
    #     sinI = np.sin(self.inclination)
    #     cosB = np.cos(self.toRadians(beta))
        
    #     """ Compute equations (107) and (108). Eq numbers may change! """
    #     cosG = (1+e*cosU)/np.sqrt(1+2*e*cosU+e**2)
    #     sinG = e*sinU/np.sqrt(1+2*e*cosU+e**2)
        
    #     """ Compute equations (117) and (118). Eq numbers may change!  """
    #     wDcosG = wC*(1+e*cosU)**2
    #     wDsinG = e*wC*(1+e*cosU)*sinU
    #     wD = np.sqrt(wDcosG**2 + wDsinG**2)
        
    #     """ Compute equation (75) """
    #     wF = wDcosG - w*cosI
        
    #     """ Compute equation (79). s2, s2F can be plus/minus depending on look
    #         direction. wwQ is the denominator of (78,79) multiplied by
    #         w^2 """
    #     wwQ = (w*sinI*cosB)**2 + wF**2
    #     s2 = look*w*np.sqrt((1-v**2)*wwQ - (w*v*sinG)**2)/wwQ
    #     s2F = look*wF*np.sqrt((1-v**2)*wwQ - (w*v*sinG)**2)/wwQ
        
    #     """ Compute the overall result (81)"""
    #     uTCN = np.array([sinG*v*(wDcosG*wF/wwQ - 1) + s2*cosG*cosB*sinI,
    #                      -w*v*wDsinG*sinI*cosB/wwQ + s2F,
    #                      v*(cosG + wDsinG*sinG*wF/wwQ) + s2*sinG*cosB*sinI])
    #     uTCN /= np.linalg.norm(uTCN)
        
    #     """ Compute the TCN vectors """
    #     TCN = self.computeTCN(beta)
        
        
    #     aTCN = np.array([wD - w*cosG*cosI,
    #                      -w*cosB*sinI,
    #                      -w*sinG*cosI])
    #     aTCN = aTCN/np.linalg.norm(aTCN)
        
    #     """ Compute the eTCN vector """
    #     eTCN = np.cross(uTCN, aTCN)
    #     eTCN/= np.linalg.norm(eTCN)
        
    #     """ Compute the aeuTCN matrix """
    #     aeuTCN = np.stack((aTCN, eTCN, uTCN), axis=1)
        
    #     """ compute the aeuI matrix """
    #     aeuI = TCN.dot(aeuTCN)
        
    #     """ Return the aeuI and aeuTCN matrices """
    #     return aeuI, aeuTCN

    
    def computeEold(self, beta, v):
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
        
        U = self.toRadians(beta) - self.arg_perigee
        e = self.e
        C = 1.0/self.planet.w*np.sqrt(self.planet.GM/(self.a*(1-e))**3)
        cosU = np.cos(U)
        sinU = np.sin(U)
        cosI = np.cos(self.inclination)
        sinI = np.sin(self.inclination)
        cosB = np.cos(self.toRadians(beta))
        
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
            
        Notes
        -----
        The calculation coded here multiples the equations in the notes
        by the planet angular velocity to allow the computation of equation
        (38) even in the case the planet angular velocity is low or zero.

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
        beta : float (Radians)
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
        beta : float
            The time in seconds since the time of the ascending node at
            which to compute the orbit angle.

        Returns
        -------
        The orbit angle.

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

    def rotateUnit(self, angles, uvec, rvec):
        """
        Rotate a unit vector around some axis for a given set of angles.
        
        This function will use the quaternion rotation formalism
        to rotate a given vector around another given vector.

        Parameters
        ----------
        angles : `numpy.ndarray`
            An array of angles to rotate by.
        uvec : `numpy.ndarray` (3,)
            The vector to rotate.
        rvec : `numpy.ndarray` (3,)
            The vector around which to rotate.

        Returns
        -------
        'numpy.ndarray` (angles.shape, 3).
            The rotated vector for each angle

        """
        
        """ Make sure we are using radians """
        cAngles = np.cos(self.toRadians(angles/2))
        sAngles = np.sin(self.toRadians(angles/2))
        
        """ Create the quaternions """
        urvec = rvec/np.linalg.norm(rvec)
        q = np.array([np.quaternion(c, *(s*urvec)) 
                      for c,s in zip(cAngles, sAngles)])
        
        """ Define the rotation vector as a quaternion """
        p = np.quaternion(0, *uvec)
        
        """ Do the computation to rotate """
        return quaternion.as_vector_part(((q*p)*q.conj()))
    
    def pointingError(self, AEU, alpha, epsilon, tau):
        """
        Rotate the basis vectors contained in the columns of AEU by the 
        pointing errors.
        
        Rotate the AEU vectors according to pointing error by using rotation 
        matrices

        Parameters
        ----------
        AEU : `numpy.ndarray` (3,3)
            Basis vectors for Azimuth, Elevation and Look direction. Each 
            column corresponds to a respective basis vector.
        alpha : float
            Azimuth error (Radians).
        epsilon : float
            Elevation error (Radians).
        tau : float
            Tilt error (Radians).

        Returns
        -------
        `numpy.ndarray` (3,3).
            The rotated basis vectors

        """
        
        cosA = np.cos(self.toRadians(alpha))
        sinA = np.sin(self.toRadians(alpha))
        cosE = np.cos(self.toRadians(epsilon))
        sinE = np.sin(self.toRadians(epsilon))
        cosT = np.cos(self.toRadians(tau))
        sinT = np.sin(self.toRadians(tau))
        
        Me = np.array([[1, 0,        0],
                       [0, cosE, -sinE],
                       [0, sinE,  cosE]])
           
        Ma = np.array([[cosA,  0, sinA],
                       [0,     1,    0],
                       [-sinA, 0, cosA]])
        
        Mt = np.array([[cosT, -sinT, 0],
                       [sinT, cosT,  0],
                       [0,    0,     1]])
        
        return AEU.dot(Me).dot(Ma).dot(Mt)
          
    def pointingErrors(self, AEU, alpha, epsilon, tau):
        """
        Rotate the basis vectors contained in the columns of AEU by the 
        pointing errors.
        
        Rotate the AEU vectors according to pointing error by using rotation 
        matrices. This function takes a list of angles by which to rotate.

        Parameters
        ----------
        AEU : `numpy.ndarray` (3,3)
            Basis vectors for Azimuth, Elevation and Look direction. Each 
            column corresponds to the respective basis vector.
        alpha : float
            Azimuth error (Radians).
        epsilon : float
            Elevation error (Radians).
        tau : float
            Tilt error (Radians).

        Returns
        -------
        `numpy.ndarray` (3,3).
            The rotated basis vectors

        """
        
        cosA = np.cos(self.toRadians(alpha))
        sinA = np.sin(self.toRadians(alpha))
        cosE = np.cos(self.toRadians(epsilon))
        sinE = np.sin(self.toRadians(epsilon))
        cosT = np.cos(self.toRadians(tau))
        sinT = np.sin(self.toRadians(tau))
        
        Me = np.array([[[1, 0, 0],
                       [0, cE, -sE],
                       [0, sE,  cE]] for cE,sE in zip(cosE, sinE)])

        Ma = np.array([[[cA, 0, sA],
                        [0, 1, 0],
                        [-sA, 0, cA]] for cA,sA in zip(cosA, sinA)])

        Mt = np.array([[[cT, -sT, 0],
                       [sT, cT,  0],
                       [0,    0,     1]] for cT,sT in zip(cosT, sinT)])
        
        """ Values are computed by using the Einstein summation convetion.
        This is a fast, intuitive way to handle the multiple indeces in the 
        ndarray. See the numpy docs for more information."""
        return np.einsum('ij,ejl,aln,tnk -> aetik', AEU, Me, Ma, Mt)
    
    def dopCen(self, 
               aeuPe, 
               off_boresight, 
               VP, 
               wavelength):
        """
        Compute the Doppler centroid at a given off-nadir (in elevation)
        angle
        
        Compute the Doppler centroid for a given off-nadir angle. The 
        off-nadir angle is the right-handed rotation of the actual look
        vector around the azimuth axis, the first column of aeuPe

        Parameters
        ----------
        aeuPe : `numpy.ndarray`, (nA, nE, nT, 3, 3)
            The actual Azimuth, Elevation, Look direction (AEU) vectors as
            columns. These are stacked by a vector of nA azimuth, nE elevation
            and nT tilt angles by which the actual aeuPe vectors are rotated
            relative the desired aeuP vectors. These are all referenced to the
            Planetary (P) reference frame.
        off_boresight : float
            The anlge by which to rotate the look vector around the actual
            azimuth axis.
        VP : `np.ndarray`, (3,)
            The satellite velocity vector in the Planetary reference frame (P).
        wavelength : float
            The carrier wavelength (m).

        Returns
        -------
        `numpy.ndarray`, (nA, nE, nT)
            The Doppler centroid for each azimuth, elevation and tilt angle
            error.

        """
        
        v = np.array([0, 
                      -np.sin(self.toRadians(off_boresight)),
                      np.cos(self.toRadians(off_boresight))])
        
        r2 = np.matmul(aeuPe, v)
        return -2*np.matmul(r2, VP)/wavelength


    def dopCens(self,
                aeuPe, 
                vAngles, 
                VP, 
                wavelength):
        """
        Compute the Doppler centroid at a given set of off-nadir (in elevation)
        angles
        
        Compute the Doppler centroid for a given set of off-nadir angles. The 
        off-nadir angle is the right-handed rotation of the actual look
        vector around the azimuth axis, the first column of aeuPe

        Parameters
        ----------
        aeuPe : `numpy.ndarray`, (nA, nE, nT, 3, 3)
            The actual Azimuth, Elevation, Look direction (AEU) vectors as
            columns. These are stacked by a vector of nA azimuth, nE elevation
            and nT tilt angles by which the actual aeuPe vectors are rotated
            relative the desired aeuP vectors. These are all referenced to the
            Planetary (P) reference frame.
        off_boresight : `numpy.ndarray`
            Array of anlges by which to rotate the look vector around the 
            actual azimuth axis.
        VP : `np.ndarray`, (3,)
            The satellite velocity vector in the Planetary reference frame (P).
        wavelength : float
            The carrier wavelength (m).

        Returns
        -------
        `numpy.ndarray`, (nA, nE, nT)
            The Doppler centroid for each azimuth, elevation and tilt angle
            error.

        """
        
        v = np.array([[0, 
                      -np.sin(self.toRadians(ob)),
                      np.cos(self.toRadians(ob))] for ob in vAngles])
        
        """ Values are computed by using the Einstein summation convetion.
        This is a fast, intuitive way to handle the multiple indeces in the 
        ndarray. See the numpy docs for more information."""
        return 2*np.einsum('ijklm,nm,l->ijkn', aeuPe, v, VP)/wavelength
        
    def PRYfromRotation(self, R, R0 = np.eye(3)):
        """
        Compute Pitch, roll and Yaw from rotation matrix
        
        This function computes the Pitch angle, Roll angle
        and Yaw angle from a given rotation matrix. The
        algorithm for computing these values can be found
        in my notes.
        
        The pitch, roll and yaw angles are given by \theta_p,
        \theta_r and \theta_y, respectively.
        
        The Pitch matrix is given by
        
        M_p = [cos\theta_p  sin\theta_p  0]
              [-sin\theta_p cos\theta_p  0]
              [0            0            1].
              
        
        The roll matrix is given by
        
        M_r = [cos\theta_r  0 -sin\theta_r]
              [0            1            0]
              [sin\theta_r  0  cos\theta_r].
        
        
        And the yaw matrix is given by
        
        M_y = [1            0            0]
              [0  cos\theta_y  sin\theta_y]
              [0  -sin\theta_y cos\theta_y].
              
        The **rotated** basis vectors i_1, j_1 and k_1 relate to the
        original basis vectors, i_0, j_0, k_0 through
        
        [i_1, j_1, k_1] = [i_0, j_0, k_0]M_p^TM_r^TM_y^T
        
        where [i_1, j_1, k_1] and [i_0, j_0, k_0] are 3x3 matrices with columns
        corresponding to the indicated basis vectors.
        
        Assuming that [i_0, j_0, k_0] is the identity matrix, this function
        takes input as R = [i_1, j_1, k_1] and computes M_p, M_r and M_y and
        the corresponding Euler angles.
        
        Parameters
        ----------
        R : `numpy.ndarray, [3,3]`
            The rotation matrix to express as pitch roll yaw.
        R0 : `numpy.ndarray, [3,3]`
            Initial basis vectors. This is defines the default from which
            pitch roll and yaw are computed. Default is the identity matrix.
    
        Returns
        -------
        dict
            A dictionary with the Euler angles in radians and the corresponding
            rotation matrices.
    
        """
        
        
        """ Compute the yaw angle """
        R = (R0.T).dot(R)
        theta_y = np.arctan2(R[2,1], R[2,2])
        c_y = np.cos(theta_y)
        s_y = np.sin(theta_y)
        
        M_y = np.array([
                        [1,0,0],
                        [0, c_y, s_y],
                        [0,-s_y, c_y]
                       ])
        
        R = R.dot(M_y)
        #print(R)
        
        """ Compute the roll angle """
        theta_r = np.arctan2(-R[2,0], R[2,2])
        c_r = np.cos(theta_r)
        s_r = np.sin(theta_r)
        
        M_r = np.array([
                        [c_r,0,-s_r],
                        [0,1,0],
                        [s_r,0,c_r]
                       ])
        
        R = R.dot(M_r)
        #print(R)
        
        """ Compute the pitch angle """
        theta_p = np.arctan2(R[1,0], R[1,1])
        c_p = np.cos(theta_p)
        s_p = np.sin(theta_p)
        
        M_p = np.array([
                        [c_p, s_p,0],
                        [-s_p,c_p,0],
                        [0,0,1]
                       ])
        
        R = R.dot(M_p)
        #print(R)
        
        return (theta_p, theta_r, theta_y), (M_p, M_r, M_y)
    
    def aeuAnglesAAEUfromDAEU(self, AAEU, DAEU):
        (ep, az, tau), (Meps, Mazi, Mtau) = self.YRPfromRotation(AAEU, DAEU)
        return (az, ep, tau), (Mazi, Meps, Mtau)
    
    def aeuAnglesDAEUfromAAEU(self, DAEU, AAEU):
        (tau, az, ep), (Mtau, Mazi, Meps) = self.PRYfromRotation(DAEU, AAEU)
        return (-az, -ep, -tau), (Mazi.T, Meps.T, Mtau.T)
    
    def YRPfromRotation(self, R, R0 = np.eye(3)):
        """
        Compute Pitch, roll and Yaw from rotation matrix
        
        This function computes the Pitch angle, Roll angle
        and Yaw angle from a given rotation matrix. The
        algorithm for computing these values can be found
        in my notes.
        
        The pitch, roll and yaw angles are given by \theta_p,
        \theta_r and \theta_y, respectively.
        
        The Pitch matrix is given by
        
        M_p = [cos\theta_p  sin\theta_p  0]
              [-sin\theta_p cos\theta_p  0]
              [0            0            1].
              
        
        The roll matrix is given by
        
        M_r = [cos\theta_r  0 -sin\theta_r]
              [0            1            0]
              [sin\theta_r  0  cos\theta_r].
        
        
        And the yaw matrix is given by
        
        M_y = [1            0            0]
              [0  cos\theta_y  sin\theta_y]
              [0  -sin\theta_y cos\theta_y].
              
        The **rotated** basis vectors i_1, j_1 and k_1 relate to the
        original basis vectors, i_0, j_0, k_0 through
        
        [i_1, j_1, k_1] = [i_0, j_0, k_0]M_p^TM_r^TM_y^T
        
        where [i_1, j_1, k_1] and [i_0, j_0, k_0] are 3x3 matrices with columns
        corresponding to the indicated basis vectors.
        
        Assuming that [i_0, j_0, k_0] is the identity matrix, this function
        takes input as R = [i_1, j_1, k_1] and computes M_p, M_r and M_y and
        the corresponding Euler angles.
        
        Parameters
        ----------
        R : `numpy.ndarray, [3,3]`
            The rotation matrix to express as pitch roll yaw.
        R0 : `numpy.ndarray, [3,3]`
            Initial basis vectors. This is defines the default from which
            pitch roll and yaw are computed. Default is the identity matrix.
    
        Returns
        -------
        dict
            A dictionary with the Euler angles in radians and the corresponding
            rotation matrices.
    
        """
        
        
        """ Compute the yaw angle """
        R = (R0.T).dot(R)
        theta_p = np.arctan2(-R[0,1], R[0,0])
        c_p = np.cos(theta_p)
        s_p = np.sin(theta_p)
        
        M_p = np.array([
                        [c_p,  s_p,  0.0],
                        [-s_p, c_p,  0.0],
                        [0.0,  0.0,  1.0]
                        ])
        R = R.dot(M_p)
        #print(R)
        
        """ Compute the roll angle """
        theta_r = np.arctan2(R[0,2], R[0,0])
        c_r = np.cos(theta_r)
        s_r = np.sin(theta_r)
        
        M_r = np.array([
                        [c_r,0,-s_r],
                        [0,1,0],
                        [s_r,0,c_r]
                       ])
        
        R = R.dot(M_r)
        #print(R)
        
        """ Compute the pitch angle """
        theta_y = np.arctan2(-R[1,2], R[1,1])
        c_y = np.cos(theta_y)
        s_y = np.sin(theta_y)
        
        M_y = np.array([
                        [1,0,0],
                        [0, c_y, s_y],
                        [0,-s_y, c_y]
                       ])
        
        R = R.dot(M_y)
        #print(R)
        
        return (theta_y, theta_r, theta_p), (M_y, M_r, M_p)