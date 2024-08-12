# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:02:00 2022

@author: ishuwa.sikaneta
"""

from measurement.measurement import state_vector
import numpy as np
from scipy.constants import c
from space.planets import earth

#%%
class perspective:
    """
    Class to compute the viewing perspective
    
    This class allows computation of the viewing persepective of the 
    satellite to a point or points on the ground. The constructor
    for this class takes and optional argument of the planet name so that
    the geometry can be defined for those bodies defined in 
    `space.planets`
    
    To-do
    -----
    
    [ ] Update to include the computeImagingGeometry functionality
    [ ] Allow the user to supply a normal vector for computation of incidence
        angles
        
    Methods
    -------
    
    setSatPosVel
        Set the satellite position and velocity vector. This can also be done
        in the constructor
    setSatPosVelFromZeroDoppler
        Set the satellite position and velocity vector from a supplied state
        vector object, a ground position, and a time around which to compute 
        the time and state vector when the satellite velocity vector is 
        perpendicular to the range vector to the target  - the zero-Doppler 
        condition
    surfaceNormal
        Compute the surface normal at a given ground point
    pointFrame
        Compute a reference frame around a supplied point. The reference frame
        is defined as a [u_lon, u_lat, u_s] where these unit vectors are in
        the direction of the local longitude, the local latitude and the 
        surface normal, respectively
    pointGeometry
        For the state vector set with setSatPosVel, this function computes, 
        for a supplied ground point, the range vector, the look angle, the
        incidence angle and the bearing angle
    computeGeometry
        For the state vector set with setSatPosVel and a supplied list of 
        off-nadir or look angles, this function computes the zero-Doppler
        ranges, incidence angles, bearing angles and an approximation to the
        ground range for surface points defined by the look angle and the 
        zero-Doppler condition
    tcn
        Static function to compute the tcn frame in the frame of the supplied
        state vector
        
    """
    
    def __init__(self, sPosVel, sTime, planet = earth()):
        self.sv = state_vector(planet = planet)
        self.setSatPosVel(sPosVel, sTime)
    
    def setSatPosVel(self, sPosVel, sTime):
        """
        Set the satellite position and velocity
        
        Set the satellite position and velocity vectors from the supplied
        state vector

        Parameters
        ----------
        sPosVel : `np.array(6,)`
            Satellite position and velocity vector.
        sTime : TYPE, optional
            The time coordinate of the state vector. The default is None.

        Returns
        -------
        None.

        """
        
        self.sPosVel = sPosVel
        self.sTime = sTime
        self.T, self.C, self.N = self.tcn(sPosVel)
        self.satLat, self.satLong, self.satHAE = self.sv.xyz2polar(sPosVel)
        rVecNadir = self.sv.computeRangeVectorsU(sPosVel, self.N)
        self.SubSatPt = sPosVel[:3] + rVecNadir
        
    def setSatPosVelFromZeroDoppler(self, sv, xG, eta0):
        """
        Set the satellite position and velocity from zero-Doppler
        
        
        Set the satellite position and velocity vector from a supplied state
        vector object, a ground position, and a time around which to compute 
        the time and state vector when the satellite velocity vector is 
        perpendicular to the range vector to the target  - the zero-Doppler 
        condition

        Parameters
        ----------
        sv : `measurement.state_vector`
            State vector object to propagate to the zero-Doppler condition.
        xG : `np.ndarray(3,)`
            The ground position of the target of interest.
        eta0 : `numpy.datetime64`
            A starting point in time for searching for the zero-Doppler
            condition.

        Returns
        -------
        None.

        """
        
        svtime, svdata, err = sv.computeBroadsideToX(eta0, xG)
        self.setSatPosVel(svdata, svtime)
        
    def surfaceNormal(self, Xg):
        """
        Compute the surface normal for a point on the surface
        
        Compute the surface normal for a point on the surface. This function
        is not static because it relies on self.sv

        Parameters
        ----------
        Xg : `np.ndarray(3,)`
            Position vector of point on the surface.

        Returns
        -------
        `np.ndarray(3,)`
            The unit surface normal vector.

        """
        
        lat, lon, _ = self.sv.xyz2SphericalPolar(Xg)
        clat = np.cos(np.radians(lat))
        slat = np.sin(np.radians(lat))
        clon = np.cos(np.radians(lon))
        slon = np.sin(np.radians(lon))
        
        n = np.array([self.sv.planet.b*clat*clon,
                      self.sv.planet.b*clat*slon,
                      self.sv.planet.a*slat])
        
        return n/np.linalg.norm(n)
    
    def pointFrame(self, Xg):
        """
        Define a planet-fixed reference frame around a surface point
        
        This frame is defined by unit vectors in the local longitude, latitude
        and surface normal directions.

        Parameters
        ----------
        Xg : `np.ndarray(3,)`
            Position vector of surface point in body-fixed frame.

        Returns
        -------
        `np.ndarray(3,3)`
            Reference frame for the point with the first column corresponding
            to a unit vector in the longitude direction, the second column
            to a unit vector in the latitude direction, and the thrid column
            to the surface normal.

        """
        
        lat, lon, _ = self.sv.xyz2SphericalPolar(Xg)
        clat = np.cos(np.radians(lat))
        slat = np.sin(np.radians(lat))
        clon = np.cos(np.radians(lon))
        slon = np.sin(np.radians(lon))
        norm = np.sqrt(self.sv.planet.a**2*slat**2 + 
                       self.sv.planet.b**2*clat**2)
        
        u_lat = np.array([-self.sv.planet.a*slat*clon,
                          -self.sv.planet.a*slat*slon,
                          self.sv.planet.b*clat])/norm
        
        u_lon = np.array([-slon,
                          clon,
                          0])
        
        u_s = np.array([self.sv.planet.b*clat*clon,
                        self.sv.planet.b*clat*slon,
                        self.sv.planet.a*slat])/norm
        
        
        return np.array([u_lon, u_lat, u_s]).T
    
    def pointGeometry(self, Xg):
        """
        Compute the angles from the satellite to a point on the surface
        
        This function returns the vector between the satellite and the 
        surface point as well as angles defined in the tcn and point reference
        frames. These angles include the look angle, incidence angle and 
        bearing angle.
        
        All quantities are computes to the satellite state vector as defined 
        by self.sPosVel

        Parameters
        ----------
        Xg : `np.ndarray(3,)`
            Position vector of the surface point.

        Returns
        -------
        rvec : `np.ndarray(3,)`
            Vector from the surface point to the satellite.
        look : `float`
            The look angle in degrees.
        incidence : `float`
            The incidence angle in degrees.
        bearing : `float`
            The bearing angle in degrees.

        """
        
        """ Compute the range vector """
        rvec = Xg - self.sPosVel[:3]
        rhat = rvec/np.linalg.norm(rvec)
        
        """ Compute the local point frame """
        pFrame = self.pointFrame(Xg)
        
        """ Compute the representation of the look vector in the pFrame """
        rhat_pFrame = (-rhat).dot(pFrame)
        
        """ Compute the imaging angles """
        look = np.degrees(np.arccos(rhat.dot(self.N)))
        incidence = np.degrees(np.arccos(rhat_pFrame[-1]))
        bearing = np.degrees(np.arctan2(rhat_pFrame[0], rhat_pFrame[1]))
        
        return rvec, look, incidence, bearing
    
    def computeSurfaceDistance(self, Xg):
        Xgl = np.vstack((self.SubSatPt, Xg[0:-1]))
        lnorm = np.linalg.norm(Xgl, axis=1)
        rnorm = np.linalg.norm(Xg, axis=1)
        ldotr = np.sum(Xgl*Xg,axis=1)
        return np.sqrt(lnorm*rnorm)*np.arccos(ldotr/lnorm/rnorm)
        
    def computeGeometry(self, off_nadir):
        """
        Compute range and angles for a series of off-nadir angles
        
        

        Parameters
        ----------
        off_nadir : iterable of angles (N)
            Iterable of off-nadir angles in radians.

        Returns
        -------
        ranges : 'np.ndarray(N,)'
            Ranges to ground points.
        incidence : 'np.ndarray(N,)'
            Incidence angles to ground points.
        bearing : 'np.ndarray(N,)'
            Bearing angles to ground points.
        'np.ndarray(N,)'
            Signed ground swath range, with zero corresponding to the
            nearest ground range. Sign is the same as for look angle

        """
        
        """ Compute the look directions """
        uhats = np.array([np.cos(eang)*self.N + np.sin(eang)*self.C 
                          for eang in off_nadir])

        """ Calculate range vectors """
        rangeVectors = self.sv.computeRangeVectorsU(self.sPosVel, uhats)

        """ Calculate the ranges """
        ranges = np.linalg.norm(rangeVectors, axis=1)

        """ Calculate the ground points """
        Xg = np.array([self.sPosVel[:3] + rV for rV in rangeVectors])
        
        """ Compute the angles """
        geometry = [self.pointGeometry(x) for x in Xg]
        incidence = [g[2] for g in geometry]
        bearing = [g[-1] for g in geometry]
        
        """ Compute the ground swath """
        Xgswath = self.computeSurfaceDistance(Xg).cumsum()
        
        return {"range": ranges,
                "off_nadir": np.degrees(off_nadir),
                "incidence": incidence, 
                "bearing": bearing, 
                "swath": Xgswath} 
    
    @staticmethod
    def tcn(svdata):
        N = -svdata[:3]/np.linalg.norm(svdata[:3])
        T = svdata[3:]/np.linalg.norm(svdata[3:])
        C = np.cross(N, T)
        C = C/np.linalg.norm(C)
        N = np.cross(T, C)
        N = N/np.linalg.norm(N)
        return T, C, N
        
#%% Define the range of elevation angles to look at
def computeGeometry(sv, off_nadir, idx = 0):
    svdata = sv.measurementData[idx]
    T,C,N = tcn(svdata)

    uhats = np.array([np.cos(eang)*N + np.sin(eang)*C for eang in off_nadir])

    # Calculate range vectors
    rangeVectors = sv.computeRangeVectorsU(svdata, uhats)

    # Calculate the ranges
    ranges = np.linalg.norm(rangeVectors, axis=1)
    rhat = rangeVectors*np.tile(1/ranges, (3,1)).T

    # Calculate the ground points
    xG = np.tile(svdata[:3], (len(off_nadir),1)) + rangeVectors
    snorm = np.array([surfaceNormal(x) for x in xG])
    
    geometry = [pointGeometry(sv, svdata, x, n) for x,n in zip(xG,snorm)]
    incidence = [g[2] for g in geometry]
    bearing = [g[-1] for g in geometry]
    
    sgn = np.sign(np.sum(np.cross(T,snorm)*rhat, axis=1))
    xGSwath = -sgn*np.insert(np.linalg.norm(xG[1:,:] - xG[:-1,:], axis=1).cumsum(), 0, 0,0)
    
    return ranges, incidence, bearing, xGSwath

#%% Define the range of elevation angles to look at
def getTiming(sv, off_nadir, idx = 0):
    svdata = sv.measurementData[idx]
    eta = sv.measurementTime[idx]
    T,C,N = tcn(svdata)

    uhats = np.array([np.cos(eang)*N + np.sin(eang)*C for eang in off_nadir])

    # Calculate range vectors
    rangeVectors = sv.computeRangeVectorsU(svdata, uhats)

    # Calculate the ranges
    ranges = np.linalg.norm(rangeVectors, axis=1)
    rhat = rangeVectors*np.tile(1/ranges, (3,1)).T
    
    # Calculate the times
    tau = 2*ranges/c

    # Calculate the ground points
    xG = np.tile(svdata[:3], (len(off_nadir),1)) + rangeVectors
    snorm = np.array([surfaceNormal(x) for x in xG])
    
    geometry = [pointGeometry(sv, svdata, x, n) for x,n in zip(xG,snorm)]
    ranges = [np.linalg.norm(g[0]) for g in geometry]
    look = [g[1] for g in geometry]
    incidence = [g[2] for g in geometry]
    bearing = [g[-1] for g in geometry]
    
    sgn = np.sign(np.sum(np.cross(T,snorm)*rhat, axis=1))
    xGSwath = -sgn*np.insert(np.linalg.norm(xG[1:,:] - xG[:-1,:], axis=1).cumsum(), 0, 0,0)
    
    inc = np.degrees(sgn*np.arccos(-np.sum(snorm*rhat, axis=1)))
    
    return ranges, rhat, inc, tau, xGSwath

#%%
def surfaceNormal(XG, sv = state_vector()):
    lat, lon, _ = sv.xyz2SphericalPolar(XG)
    clat = np.cos(np.radians(lat))
    slat = np.sin(np.radians(lat))
    clon = np.cos(np.radians(lon))
    slon = np.sin(np.radians(lon))
    
    n = np.array([sv.planet.b*clat*clon,
                  sv.planet.b*clat*slon,
                  sv.planet.a*slat])
    
    return n/np.linalg.norm(n)

#%%
def pointFrame(XG, sv = state_vector()):
    lat, lon, _ = sv.xyz2SphericalPolar(XG)
    clat = np.cos(np.radians(lat))
    slat = np.sin(np.radians(lat))
    clon = np.cos(np.radians(lon))
    slon = np.sin(np.radians(lon))
    norm = np.sqrt(sv.planet.a**2*slat**2 + sv.planet.b**2*clat**2)
    
    u_lat = np.array([-sv.planet.a*slat*clon,
                      -sv.planet.a*slat*slon,
                      sv.planet.b*clat])/norm
    
    u_lon = np.array([-slon,
                      clon,
                      0])
    
    u_s = np.array([sv.planet.b*clat*clon,
                    sv.planet.b*clat*slon,
                    sv.planet.a*slat])/norm
    
    
    return np.array([u_lon, u_lat, u_s]).T

#%%
def satSurfaceGeometry(sv, off_nadir, azi_angle):
    soff_nadir = np.sin(off_nadir)
    coff_nadir = np.cos(off_nadir)
    sazi_angle = np.sin(azi_angle)
    cazi_angle = np.cos(azi_angle)
    
    u = np.stack((soff_nadir*cazi_angle,
                  soff_nadir*sazi_angle,
                  coff_nadir), axis=1)
    
    r = np.zeros_like(azi_angle)
    incidence = np.zeros_like(azi_angle)
    rvec = np.zeros((len(azi_angle), 3), dtype=float)
    snormal = np.zeros_like(rvec)
    
    for k in range(len(azi_angle)):
        s = sv.measurementData[k]
        look = u[k]
        T = s[3:]/np.linalg.norm(s[3:])
        N = -surfaceNormal(s, sv)
        C = np.cross(N,T)
        tcn_look = np.stack((T,C,N), axis=1).dot(look) 
        XG = sv.computeGroundPositionU(s, tcn_look)
        rvec[k,:] = s[0:3] - XG
        r[k] = np.linalg.norm(rvec[k,:])
        rhat = rvec[k,:]/r[k]
        snormal[k] = surfaceNormal(XG, sv)
        incidence[k] = np.arccos(rhat.dot(snormal[k]))
        
    return rvec, snormal, r, incidence


#%% New find nearest based on ordered time
def findNearest(timeArray, eta):
    nElements = len(timeArray)
    def new_bracket(bracket):
        center = int((bracket[1] + bracket[0])/2)
        if timeArray[center] < eta:
            return center, bracket[1]
        else:
            return bracket[0], center
    
    bracket = new_bracket([0, nElements])
    while bracket[1] - bracket[0] > 1:
        bracket = new_bracket(bracket)
        
    if (np.abs(eta - timeArray[bracket[0]]) < 
        np.abs(eta - timeArray[bracket[1]])):
        return bracket[0]
    else:
        return bracket[1]

#%% Function to get body-fixed TCN frame from state vector
def tcn(svdata):
    N = -svdata[:3]/np.linalg.norm(svdata[:3])
    T = svdata[3:]/np.linalg.norm(svdata[3:])
    C = np.cross(N, T)
    C = C/np.linalg.norm(C)
    N = np.cross(T, C)
    N = N/np.linalg.norm(N)
    return T, C, N

#%%
def computeImagingGeometry(sv, eta, xG, xG_snormal = None):
    if xG_snormal is None:
        xG_snormal = surfaceNormal(xG, sv)
        
    idx = findNearest(sv.measurementTime, eta)
    
    mysv = state_vector(planet=sv.planet)
    mysv.add(sv.measurementTime[idx], sv.measurementData[idx])
    
    svtime, svdata, err = mysv.computeBroadsideToX(eta, xG)
    
    rvec, look, incidence, azimuth, bearing = pointGeometry(sv,
                                                            svdata,
                                                            xG,
                                                            xG_snormal)
    return rvec, incidence, azimuth, bearing, [svtime, svdata], err

#%%
def pointGeometry(sv,
                  satStateVector,
                  groundXYZ,
                  groundNormal):
    """ Compute the range vector """
    rvec = groundXYZ - satStateVector[:3]
    rhat = rvec/np.linalg.norm(rvec)
    
    """ Compute the local TCN frame """
    T,C,N = tcn(satStateVector)
    
    """ Compute the local point frame """
    pFrame = pointFrame(groundXYZ, sv)
    
    """ Compute the representation of the look vector in the pFrame """
    rhat_pFrame = (-rhat).dot(pFrame)
    
    """ Compute the imaging angles """
    look = np.degrees(np.arccos(rhat.dot(N)))
    incidence = np.degrees(np.arccos(rhat_pFrame[-1]))
    azimuth = np.degrees(np.arctan2(rhat_pFrame[1], rhat_pFrame[0]))
    bearing = np.mod(90-azimuth, 360)
    
    return rvec, look, incidence, azimuth, bearing
























    