import datetime
from math import asin, atan, atan2
from scipy.integrate import solve_ivp
# import numpy.matrix as mat
import xml.etree.ElementTree as etree
import numpy as np
from numpy import meshgrid, arange, sqrt, zeros, cumprod, array, diag, cos, sin
import os
import math
from timeit import default_timer as timer
from numba import njit, prange
from space.planets import earth
from types import SimpleNamespace
import bisect


class measurement:
    """
    Class to hold measurement data.
    
    The measurement data are vector in nature with an associated time. The
    two main data members of this class are, therefore, the measurementData
    and measurementTime lists. This class is suitable for such things as
    state vectors or attitude vectors.
    
    This is one of the fundamental classes of sip_tools
    
    Methods
    -------
    add
        Add a measurement vector and a measurement time
    get
        Get a measurement at a particular time
    findNearest
        Return the measurement nearest to the given time
    
    Notes
    -----
    This class is meant as a base class for more specific types of
    measurements
    
    """
    def __init__(self, mTime=[], mData=[]):
        self.reference_time = datetime.datetime.now()
        if(mTime):
            idxs = np.argsort(mTime)
            self.measurementTime = [mTime[idx] for idx in idxs]
            self.measurementData = [mData[idx] for idx in idxs]

    def add(self, mTime, mData):
        try:
            pos = bisect.bisect_left(self.measurementTime, mTime)
            if (pos == len(self.measurementTime) or 
                self.measurementTime[pos] != mTime):
                self.measurementTime.insert(pos, mTime)
                self.measurementData.insert(pos, np.array(mData))
        except AttributeError:
            self.measurementTime = []
            self.measurementData = []
            self.measurementTime.append(mTime)
            self.measurementData.append(np.array(mData))
            
            
    def get(self, mTime):
        # return the measurement or estimate at the given time
        for cT, cM in zip(self.measurementTime, self.measurementData):
            if cT > mTime:
                return cM
        print("Did not find a time")
        return self.measurementData[-1]

    def findNearest(self, dtime):
        if type(self.measurementTime[0]) == datetime.datetime:
            d_array = np.array([(dtime - t).total_seconds() 
                for t in self.measurementTime])
            return np.argmin(np.abs(d_array))
        elif type(self.measurementTime[0]) == np.datetime64:
            d_array = np.array([(dtime - t)/np.timedelta64(1,'s') 
                for t in self.measurementTime])
            return np.argmin(np.abs(d_array))
        
        return 0
    
class state_vector(measurement):
    """
    Class to represent state vectors
    
    This is a fundamental class for radar signal processing. It allows for
    propagation of state vectors using numerical integration of the harmonics 
    spherical harmonic constants
    
    Methods
    -------
    xyz2polar
        Convert ECEF Cartesian coordinates to polar coordinates. This is an
        iterative method
    xyz2SphericalPolar
        Convert ECEF Cartesian coordinates to spherical polar coordinates. As
        opposed to the function above, which converts to polar coordinates on
        the ellipsoid, this method converts to polar coordinates on a sphere.
    llh2xyz
        Convert lat, long, height to ECEF Cartesian coordinates
    diffG
        Return the differential geometry constants associated with a given
        expanded state vector
    computeBroadsideToX
        Compute the time at which a point X is perpendicular to the velocity
        vector
    getDateTimeXML
        Compute a datetime from an XML node
    loadSphericalHarmonics
        Load shperical harmonic coefficients from a file
    harmonicsPotential
        Calculate the harmonicspotential
    ellipsoidPotential
        Calculate the GRS80 ellipsoid potential
    harmonicsdelR
        Calculate partial derivative of harmonicsPotential with respect to R 
    harmonicsdelT
        Calculate partial derivative of harmonicsPotential with respect to T 
    harmonicsdelU
        Calculate partial derivative of harmonicsPotential with respect to U
    harmonicsdelRdelR
        Calculate second partial derivative of harmonicsPotential with respect 
        to R and R 
    harmonicsdelTdelR
        Calculate second partial derivative of harmonicsPotential with respect 
        to T and R 
    harmonicsdelUdelR
        Calculate second partial derivative of harmonicsPotential with respect 
        to U and R 
    harmonicsdelTdelT
        Calculate second partial derivative of harmonicsPotential with respect 
        to T and T 
    harmonicsdelUdelT
        Calculate second partial derivative of harmonicsPotential with respect 
        to U and T 
    harmonicsdelUdelU
        Calculate second partial derivative of harmonicsPotential with respect 
        to R and R
    expandedState
        Calculate the expanded state vector. Given a normal state vector, with
        position and velocity vectors, the acceleration and jerk are 
        calculated and appended.
    estimate
        Estimate the state vector at a given time
    estimateTimeRange
        Estimate the state vectors at a range of times
    geoidHeight
        Use harmonics coefficients to compute the geoid height
    satEQM
        Return the system of differential equations for the satellite motion
    secondDtve
        Calculate the second derivative of the variable transformation
        functions. d2r/dxdx, d2r/dxdy, d2r/dxdz, d2T/dxdx, ... See appendix
        B.2 in jerk calculation
    secondSphericalDerivative
        Calculate the Hessian matrix for harmonicspotential. This is presented
        explicitely in B.2 in the jerk calculation
    grs80radius
        Calculate the GRS80 radius at the given latitude
    legendreNorm
        Calculate the normalized legendre coefficients for the given angles
    myLegendre
        Calculate the associated Legendre coefficients for the given angles
    
    """
    Nharmonics = 20
    NharmonicsToLoad = 20
    harmonics = []
    
    
    def __init__(self, 
                 svFile=None,
                 planet = None,
                 harmonicsfile=None, 
                 harmonicsCoeff=None):
        self.reference_time = datetime.datetime.now()
        self.planet = planet or earth()
        harmonicsFile = harmonicsfile or self.planet.sphericalHarmonicsFile
        harmonicsCoeff = harmonicsCoeff or self.planet.nharmonics
        if(os.path.exists(harmonicsFile) and harmonicsCoeff):
            try:
                self.loadSphericalHarmonics(harmonicsFile, harmonicsCoeff)
            except IOError as e:
                print(e)
                
        
        if(svFile):
            print("loading the state vector file:")
            print(svFile)
            self.readStateVectors(svFile)
            
            

    def xyz2polar(self, mData, maxiter = 1000, etol=1e-9):
        """
        Convert Cartesian ECEF XYZ to geographical polar coordinates EPSG:4326
        
        Parameters
        ----------
        mData : numpy.`numpy.ndarray`, (3,) or `list`, [`float`]
            The X,Y,Z coordinates in ECEF Catersian space.
        maxiter : int
            Maximum number of iterations in the method
        etol : float
            Error bound for the iteration. Once the difference in iterations
            is small than this number, the algorithm stops
            
        Returns
        -------
        latitude : `float`
            The computed latitude.
        longitude : `float`
            The computed longitude.
        hae : `float`
            The computed height above ellipsoid.
            
        Notes
        -----
        This implementation is faster than the method defined in the class
        satGeometry
        
        """
        a = self.planet.a
        b = self.planet.b
        f=(a-b)/a
        es=2.0*f-f*f;
        p=sqrt(mData[0]*mData[0]+mData[1]*mData[1])
        latitude=atan(mData[2]*a/p*b)
        ess=(a*a-b*b)/(b*b)
        for k in range(1,maxiter):
            la = atan2(mData[2]+ess*b*sin(latitude)**3,p-es*a*cos(latitude)**3)
            if(abs(latitude-la)<etol):
                break
            latitude = la
        longitude = atan2(mData[1],mData[0])
        Nphi=a/sqrt(1.0-es*sin(latitude)**2)
        hae=p/cos(latitude)-Nphi
        return np.degrees(latitude), np.degrees(longitude), hae

    def xyz2SphericalPolar(self, mData):
        """
        Convert ECEF XYZ to spherical polar coordinates
        
        Parameters
        ----------
        mData : `numpy.ndarray` or list
            The input X,Y,Z.
            
        Returns
        -------
        latitude : `float`
            Latitude anlge.
        longitude : `float`
            Longitude anlge.
        r : `float`
            The raidus.
            
        """
        r = sqrt(mData[0]*mData[0]+mData[1]*mData[1]+mData[2]*mData[2])
        latitude = asin(mData[2]/r)
        longitude = atan2(mData[1],mData[0])
        return np.degrees(latitude), np.degrees(longitude), r
    
    def llh2xyz(self, mData):
        """
        Convert lat, long, height to ECEF Cartesian XYZ
        
        Parameters
        ----------
        mData : `numpy.ndarray` or list
            The input lat, long, height.
            
        Returns
        -------
        `numpy.ndarray`, (3,)
            The X,Y,Z coordinates in ECEF Cartesian space.
            
        """
        # mData contains the lat, lon, height in degrees
        a = self.planet.a
        b = self.planet.b
        phi = np.radians(mData[0])
        lam = np.radians(mData[1])
        h = mData[2]
        
        N = a**2/sqrt(a**2*cos(phi)**2+b**2*sin(phi)**2)
        X = (N+h)*cos(phi)*cos(lam)
        Y = (N+h)*cos(phi)*sin(lam)
        Z = (b**2*N/a**2+h)*sin(phi)
        return np.array([X,Y,Z])
    
    def computeBroadsideToLLH(self, eta, llh, maxiter = 10, etol=1e-6):
        return self.computeBroadsideToX(eta, 
                                        self.llh2xyz(llh), 
                                        maxiter = maxiter, 
                                        etol = etol)
    
    def computeBroadsideToX(self, eta, X, maxiter = 10, etol=1e-6):
        """
        Compute the broadside state vector to vector X
        
        Given a curve described with state vectors, there is a time when the
        vector between the curve c(t) and the position X is perpendicular to
        the the curve velocity vector, c'(t).
        
        Parameters
        ----------
        eta : datetime
            Initial estimate of broadside time.
        X : `numpy.ndarray`, (3,)
            The XYZ coordinates of the point of interest.
        maxiter : int
            Maximum number of iterations in the Newton method
        etol : float
            Error bound for the iteration. Once the difference in iterations
            is small than this number, the algorithm stops
            
        Returns
        -------
        `list`, [`float`, `numpy.ndarray`, (3,), `float`]
            The broadside time, the state vector at this time an the error in
            the broadside position.
            
        """
        for k in range(maxiter):
            slaveX = self.estimate(eta)
            slaveV = self.satEQM(slaveX, 0.0)
            slX = slaveX[0:3]
            slV = slaveX[3:]
            slA = slaveV[3:]
            error = np.dot(slV,slX-X)/(np.dot(slA, slX-X) + np.dot(slV, slV))
            
            # Update the time variable
            if type(eta) == datetime.datetime:
                eta = eta - datetime.timedelta(seconds=error)
            else:
                eta = eta - np.timedelta64(int(error*1e9), 'ns')
            
            # Check to break from the loop. Recall that we only measure time
            # to the closest 1.0e-6 in the usec field
            if(abs(error) < etol): 
                break
            
        return [eta, slaveX, error]
    
    #%% Code to compute the ground position given u and v
    def computeGroundPosition(self, X, u = 0, v = 0.707107, h=0):
        xhat = -X[:3]/np.linalg.norm(X[:3])
        vhat = X[3:]/np.linalg.norm(X[3:])
        
        """ Compute a vector perpendicular to both """
        what = np.cross(xhat, vhat)
        what = what/np.linalg.norm(what)
        
        """ Find the matrix to invert """
        M = np.array([xhat, vhat, what])
        Minv = np.linalg.inv(M)
        
        """ Homogeneous solution """
        y = np.array([v, u, 0])
        uhom = np.dot(Minv, y)
        
        """ Find a unit vector specific solution """
        C = uhom.dot(uhom) - 1
        B = 2*uhom.dot(what)
        tA = (-B + np.sqrt(B**2 - 4*C))/2
        tB = (-B - np.sqrt(B**2 - 4*C))/2
        uhatA = uhom + tA*what
        uhatB = uhom + tB*what
        uhat = uhatA if uhatA.dot(what) >= 0 else uhatB
        
        """ Now find where the vector hits the ground """
        Xs = X[:3]
        a = self.planet.a
        b = self.planet.b
        mXs = Xs/np.array([a+h, a+h, b+h])
        mu = uhat/np.array([a+h, a+h, b+h])
        
        mC = mXs.dot(mXs) - 1
        mB = 2*mXs.dot(mu)
        mA = mu.dot(mu)
        
        mtA = (-mB + np.sqrt(mB**2 - 4*mA*mC))/(2*mA)
        mtB = (-mB - np.sqrt(mB**2 - 4*mA*mC))/(2*mA)
        
        mt = mtA if np.abs(mtA) < np.abs(mtB) else mtB
        
        return Xs + mt*uhat
    
    #%% Code to compute the ground position given u and v
    def computeGroundPositionU(self, X, uhat, h=0):
        
        """ Find where the vector hits the ground """
        Xs = X[:3]
        a = self.planet.a
        b = self.planet.b
        mXs = Xs/np.array([a+h, a+h, b+h])
        mu = uhat/np.array([a+h, a+h, b+h])
        
        mC = mXs.dot(mXs) - 1
        mB = 2*mXs.dot(mu)
        mA = mu.dot(mu)
        
        mtA = (-mB + np.sqrt(mB**2 - 4*mA*mC))/(2*mA)
        mtB = (-mB - np.sqrt(mB**2 - 4*mA*mC))/(2*mA)
        
        mt = mtA if np.abs(mtA) < np.abs(mtB) else mtB
        
        return Xs + mt*uhat
    
    def computeRangeVectorsU(self, X, uhats, h=0):

        """ Find where the vector hits the ground """
        Xs = X[:3]
        a = self.planet.a
        b = self.planet.b
        mXs = Xs/np.array([a+h, a+h, b+h])
        De = np.array([[1/(a+h), 0, 0],[0, 1/(a+h), 0],[0, 0, 1/(b+h)]])
        mu = np.matmul(uhats, De)

        mA = np.sum(mu*mu, axis=-1)
        mB = 2*np.matmul(mu, mXs)
        mC = (mXs.dot(mXs) - 1)*np.ones_like(mA)

        mtA = (-mB + np.sqrt(mB**2 - 4*mA*mC))/(2*mA)
        mtB = (-mB - np.sqrt(mB**2 - 4*mA*mC))/(2*mA)

        msk = (np.abs(mtA) < np.abs(mtB)).astype(int)
        mt = mtA*msk + mtB*(1-msk)

        """ Compute the range vectors and return """
        return np.stack((mt, mt, mt), axis=-1)*uhats
        
    def getDateTimeXML(self, XMLDateTimeElement):
        """
        Read a datetime from an XML snippet
        
        Parameters
        ----------
        XMLDateTimeElement : `etree.Element`
            A node describing a time.
            
        Returns
        -------
        `datetime.datetime`
            A datetime representation of the time.
            
        """
        return datetime.datetime(int(XMLDateTimeElement.find('year').text),
                                 int(XMLDateTimeElement.find('month').text),
                                 int(XMLDateTimeElement.find('day').text),
                                 int(XMLDateTimeElement.find('hour').text),
                                 int(XMLDateTimeElement.find('minute').text),
                                 int(XMLDateTimeElement.find('sec').text),
                                 int(XMLDateTimeElement.find('usec').text))
    
    def loadSphericalHarmonics(self, filename, Nharmonics=None):
        """
        Load spherical harmonics from file
        
        Load spherical harmonics such as egm96 or egm2008 from a csv file
        
        Parameters
        ----------
        filename : `str`
            The filename.
        Nharmonics : `int`, optional
            The number of harmonics to read. If set to None, then the entire
            set of harmonics will be read. The default is None.
            
        Returns
        -------
        None.
        
        """
        self.harmonics = []
        self.hrmC = np.zeros((Nharmonics+1, Nharmonics+1), dtype=float)
        self.hrmS = np.zeros((Nharmonics+1, Nharmonics+1), dtype=float)

        # Determine the number of harmonics to load
        # This is a cute trick from stackoverflow. The or function
        # chooses the first argument if both true for integers
        self.NharmonicsToLoad = Nharmonics or self.NharmonicsToLoad

        try:
          f = open(filename, 'r')
        except IOError:
          print("Could not open the spherical harmonics file")
          print("Could not open: " + filename)
          print("Calculations will proceed with zero-order approximation")
          return

        row = f.readline()
        fields = row.split()
        while int(fields[0])<(self.NharmonicsToLoad+1):
            self.harmonics.append([int(fields[0]),
                                   int(fields[1]),
                                   float(fields[2]),
                                   float(fields[3])])
            self.hrmC[int(fields[0]),int(fields[1])] = float(fields[2])
            self.hrmS[int(fields[0]),int(fields[1])] = float(fields[3])
            row = f.readline()
            if(row):
                fields = row.split()
            else:
                break
        f.close()
        self.Nharmonics = self.NharmonicsToLoad
        return

    def selectEGMNew(self,idx):
        """
        Read the spherical harmonic coefficients
        
        These are the Cosine coefficients at idx and the Sine coefficients
        at idx, C_idx and S_idx as presented on the Wikipedia page on the
        `Geoid <https://en.wikipedia.org/wiki/Geoid>`_
        
        Parameters
        ----------
        idx : `int`
            The index of the desired harmonics.
            
        Returns
        -------
        `numpy.ndarray`, (N,2)
            The Cosine and Sine coefficients.
            
        """
        C = []
        S = []
        for x in self.harmonics:
            if(int(x[0])==idx):
                C.append(float(x[2]))
                S.append(float(x[3]))
        return array([C,S]).T
        
    def selectEGM(self,idx):
        C = []
        S = []
        for x in self.harmonics:
            if(int(x[0])==idx):
                C.append(float(x[2]))
                S.append(float(x[3]))
        # return mat([C,S]).H
        return np.array([C,S]).T
    
    def grs80radius(self, lat):
        """
        The GRS80 radius at a particular latitude
        
        Parameters
        ----------
        lat : `float`
            Latitude.
            
        Returns
        -------
        `float`
            The computed radius. The distance from the centre of the earth to
            a point on the surface of the ellipsoid at the given latitude.
            Answer in meters.
            
        """
        ee = (self.planet.a/self.planet.b)**2 - 1.0
        return self.planet.a/sqrt(1.0 + ee*sin(lat)**2)
        
    def geoidHeight(self, lat, lon):
        """
        Compute the geoid height at the given lat, long
        
        Compute the geoid height difference as compared to the GRS80 ellipsoid
        at the given lat, long coordinates
        
        Parameters
        ----------
        lat : `float`
            Latitude.
        lon : `float`
            Longitude.
            
        Returns
        -------
        `float`
            Difference between ellipsoid height and geoid height in m.
            
        """
        # Compute the GRS80 equivalent radius
        r = self.grs80radius(lat)
        h = -(self.harmonicsPotential(r, lat, lon) 
              - self.ellipsoidPotential(r, lat, lon))
        return h/self.harmonicsdelR(r, lat, lon)

    def _getLCS(self,
                lat,
                lon,
                nmLegendreCoeffs = None,
                cosines = None,
                sines = None):
        
        if nmLegendreCoeffs is None:
            nmLegendreCoeffs = self.myLegendre(self.Nharmonics, sin(lat))
        if cosines is None:
            cosines = np.array([cos(l*lon) 
                                for l in range (0, self.Nharmonics+1)])
        if sines is None:
            sines = np.array([sin(l*lon) 
                              for l in range (0, self.Nharmonics+1)])
            
        return nmLegendreCoeffs, cosines, sines
        
    def harmonicsPotential(self, 
                       r, 
                       lat, 
                       lon,
                       nmLegendreCoeffs = None,
                       cosines = None,
                       sines = None):
        """
        Compute the harmonics gravitational potantial at the given lat, long
        
        Compute the gravitational potential at the given spherical polar
        coordinates, (not EPSG:4326). The computation uses sperical 
        harmonics, for instance, egm96, and the formula found
        here: Geoid_. The gravitational
        acceleration at the point in space is given by the gradient of this 
        potential.
        
        Parameters
        ----------
        r : `float`
            Range from centre of the earth to point of interest (m).
        lat : `float`
            Latitude angle to point of interest (rad).
        lon : `float`
            Longitude to the point of interest (rad).
            
        Returns
        -------
        D : `float`
            The harmonics potential this point.
            
        """
        GM = self.planet.GM
        a = self.planet.a

        # Compute the coefficients for the gradient of the gravitational 
        # potential
        D = 1.0*GM/r
        
        
        if nmLegendreCoeffs is None:
            nmLegendreCoeffs = self.myLegendre(self.Nharmonics, 
                                                   sin(lat))
        if cosines is None:
            cosines = [cos(np.mod(l*lon, 2.0*np.pi)) 
                       for l in range (0, self.Nharmonics+1)]
        if sines is None:
            sines = [sin(np.mod(l*lon, 2.0*np.pi)) 
                     for l in range (0, self.Nharmonics+1)]
            
        nterm = np.arange(2, self.Nharmonics+1)
        rterm = (a/r)**nterm
        D = 1.0*GM/r*(1.0 + 
                      np.dot(rterm,
                             (nmLegendreCoeffs[2:,:]*
                              self.hrmC[2:,:]).dot(cosines)+
                             (nmLegendreCoeffs[2:,:]*
                              self.hrmS[2:,:]).dot(sines)))
        return D
    
    def ellipsoidPotential(self, r, lat, lon):
        """
        Compute the GRS80 ellipsoid potential
        
        Compute the GRS80 ellipsoid potential at the given point. The 
        potential should be more or less constant for every point on a 
        particular ellipse that contains the point of interest.
        
        Parameters
        ----------
        r : `float`
            Range from centre of the earth to point of interest (m).
        lat : `float`
            Latitude angle to point of interest (rad).
        lon : `float`
            Longitude to the point of interest (rad).
            
        Returns
        -------
        D : `float`
            The GRS80 ellipsoid potential the point.
            
        """
        GM = self.planet.GM
        a = self.planet.a
        
        J = [self.planet.J2, 
             self.planet.J4, 
             self.planet.J6, 
             self.planet.J8]
        
        # Compute the coefficients of the harmonic expansion
        coeff = [-j/sqrt(2.0*l+1.0)*self.myLegendre(l,sin(lat))[-1,0]*(a/r)**l 
                 for j,l in zip(J, range(2,10,2))]
        
        # Add the very first term to the cumulative sum and multiple by 
        # physical constants
        D = GM/r*(1.0 + sum(coeff))

        return D 

    def harmonicsdel(self, 
                 r, 
                 lat, 
                 lon,
                 nmLegendreCoeffs = None,
                 cosines = None,
                 sines = None):
        GM = self.planet.GM
        a = self.planet.a
        u = sin(lat)
        v = cos(lat)
        rSquare = r**2

        nmLegendreCoeffs, cosines, sines = self._getLCS(lat,
                                                        lon,
                                                        nmLegendreCoeffs,
                                                        cosines,
                                                        sines)
        
        nterm = np.arange(2, self.Nharmonics+1)
        mterm = np.arange(self.Nharmonics+1)
        rterm = (a/r)**nterm
        Dr = -1.0*GM/rSquare*(1.0 + 
                              np.dot(rterm*(nterm+1),
                                     (nmLegendreCoeffs[2:,:]*
                                      self.hrmC[2:,:]).dot(cosines)+
                                     (nmLegendreCoeffs[2:,:]*
                                      self.hrmS[2:,:]).dot(sines)))
        Dt = GM/r*(np.dot(rterm,
                          (nmLegendreCoeffs[2:,:]*
                           self.hrmC[2:,:]).dot(-mterm*sines)+
                          (nmLegendreCoeffs[2:,:]*
                           self.hrmS[2:,:]).dot(mterm*cosines)))
        M,N = np.meshgrid(mterm, nterm)
        M[M>N]=0
        F1 = nmLegendreCoeffs[2:,:]*u*N
        F2 = nmLegendreCoeffs[1:-1,:]*np.sqrt((N**2-M**2)*(N+0.5)/(N-0.5))
        Du = v*GM/(u**2-1.0)/r*np.dot(rterm,
                                      ((F1-F2)*
                                       self.hrmC[2:,:]).dot(cosines)+
                                      ((F1-F2)*
                                       self.hrmS[2:,:]).dot(sines))
        return Dr, Du, Dt
        
    def harmonicsdelR(self, 
                  r, 
                  lat, 
                  lon,
                  nmLegendreCoeffs = None,
                  cosines = None,
                  sines = None):
        """
        Compute the derivative of the harmonics potential with respect to r.
        
        This is the component of gravitational acceleration in the direction 
        of the vector from the centre of the earth to the point.
        
        Parameters
        ----------
        r : `float`
            Range from centre of the earth to point of interest (m).
        lat : `float`
            Latitude angle to point of interest (rad).
        lon : `float`
            Longitude to the point of interest (rad).
            
        Returns
        -------
        D : `float`
            The derivative of the harmonics potential wrt r at the point.
            
        """
        GM = self.planet.GM
        a = self.planet.a
        rSquare = r**2

        # Apply the formula
        D = -1.0*GM/rSquare

        nmLegendreCoeffs, cosines, sines = self._getLCS(lat,
                                                        lon,
                                                        nmLegendreCoeffs,
                                                        cosines,
                                                        sines)
        
        for k in range(2,self.Nharmonics+1):
            egmIndex = int((k-2)*(k+3)/2)
            mv=nmLegendreCoeffs[k,0:(k+1)]
            cs = sum([mv[l]*(cosines[l]*self.harmonics[egmIndex+l][2] 
                             + sines[l]*self.harmonics[egmIndex+l][3]) 
                      for l in range(0,k+1)])
            D += -1.0*GM*cs*(k+1.0)*(a/r)**k/rSquare

        return D 
    
    def harmonicsdelRdelR(self, 
                      r, 
                      lat, 
                      lon,
                      nmLegendreCoeffs = None,
                      cosines = None,
                      sines = None):
        """
        Compute the second derivative of the harmonics potential with respect 
        to r twice.
        
        Parameters
        ----------
        r : `float`
            Range from centre of the earth to point of interest (m).
        lat : `float`
            Latitude angle to point of interest (rad).
        lon : `float`
            Longitude to the point of interest (rad).
            
        Returns
        -------
        D : `float`
            The second derivative of the harmonics potential wrt r twice at the 
            point of interest.
            
        """
        GM = self.planet.GM
        a = self.planet.a
        rCubed = r**3

        # Apply the formula
        D = 2.0*GM/rCubed

        nmLegendreCoeffs, cosines, sines = self._getLCS(lat,
                                                        lon,
                                                        nmLegendreCoeffs,
                                                        cosines,
                                                        sines)
        
        for k in range(2,self.Nharmonics+1):
            egmIndex = int((k-2)*(k+3)/2)
            mv=nmLegendreCoeffs[k,0:(k+1)]
            cs = sum([mv[l]*(cosines[l]*self.harmonics[egmIndex+l][2] 
                             + sines[l]*self.harmonics[egmIndex+l][3]) 
                      for l in range(0,k+1)])
            D += GM*cs*(k+1.0)*(k+2.0)*(a/r)**k/rCubed

        return D 
    
    def harmonicsdeldel(self, 
                      r, 
                      lat, 
                      lon,
                      nmLegendreCoeffs = None,
                      cosines = None,
                      sines = None):
        
        GM = self.planet.GM
        a = self.planet.a
        u = sin(lat)
        v = cos(lat)
        rCubed = r**3
        nterm = np.arange(2, self.Nharmonics+1)
        mterm = np.arange(self.Nharmonics+1)
        rterm = (a/r)**nterm
        M,N = np.meshgrid(mterm, nterm)
        DSQ1 = N**2-M**2
        DSQ2 = (N-1)**2-M**2
        DSQ1[DSQ1<0] = 0
        DSQ2[DSQ2<0] = 0

        nmLegendreCoeffs, cosines, sines = self._getLCS(lat,
                                                        lon,
                                                        nmLegendreCoeffs,
                                                        cosines,
                                                        sines)
        
        SQ1 = (N**2-M**2)*(N+0.5)/(N-0.5)
        SQ2 = ((N-1)**2-M**2)*(N-0.5)/(N-1.5)
        SQ1[SQ1<0] = 0.0
        SQ2[SQ2<0] = 0.0
        
        F1 = nmLegendreCoeffs[2:,:]*u*N
        F2 = nmLegendreCoeffs[1:-1,:]*np.sqrt(SQ1)
        
        G1 = nmLegendreCoeffs[2:,:]*(u**2*N**2-N)/v**2
        G2 = nmLegendreCoeffs[1:-1,:]*np.sqrt(SQ1)*(1-N)*2*u/v**2
        G3 = nmLegendreCoeffs[0:-2,:]*np.sqrt(SQ1*SQ2)/v**2
        
        DrDr = 1.0*GM/rCubed*(2.0 + 
                              np.dot(rterm*(nterm+1)*(nterm+2),
                                     (nmLegendreCoeffs[2:,:]*
                                      self.hrmC[2:,:]).dot(cosines)+
                                     (nmLegendreCoeffs[2:,:]*
                                      self.hrmS[2:,:]).dot(sines)))
        
        DrDu = -v*GM/r**2/(u**2-1.0)*(np.dot(rterm*(nterm+1),
                                      ((F1-F2)*
                                       self.hrmC[2:,:]).dot(cosines)+
                                      ((F1-F2)*
                                       self.hrmS[2:,:]).dot(sines)))
        DrDt = -GM/r**2*(np.dot(rterm*(nterm+1),
                                (nmLegendreCoeffs[2:,:]*
                                 self.hrmC[2:,:]).dot(-mterm*sines)+
                                (nmLegendreCoeffs[2:,:]*
                                 self.hrmS[2:,:]).dot(mterm*cosines)))
        DuDu = GM/r*np.dot(rterm,
                           ((G1+G2+G3)*
                             self.hrmC[2:,:]).dot(cosines)+
                           ((G1+G2+G3)*
                             self.hrmS[2:,:]).dot(sines))    
        DuDt = v*GM/r/(u**2-1.0)*(np.dot(rterm,
                                  ((F1-F2)*
                                   self.hrmC[2:,:]).dot(-mterm*sines)+
                                  ((F1-F2)*
                                   self.hrmS[2:,:]).dot(mterm*cosines)))
        DtDt = GM/r*(np.dot(rterm,
                          (nmLegendreCoeffs[2:,:]*
                           self.hrmC[2:,:]).dot(-mterm**2*cosines)+
                          (nmLegendreCoeffs[2:,:]*
                           self.hrmS[2:,:]).dot(-mterm**2*sines)))
        
        return DrDr, DrDu, DrDt, DuDu, DuDt, DtDt
    
    # T corresponds to longitude    
    def harmonicsdelT(self, 
                  r, 
                  lat, 
                  lon,
                  nmLegendreCoeffs = None,
                  cosines = None,
                  sines = None):
        """
        Compute the derivative of the harmonics potential with respect to 
        theta.
        
        This is the component of gravitational acceleration in the direction 
        of theta. More or less east/west gravitational acceleration
        
        Parameters
        ----------
        r : `float`
            Range from centre of the earth to point of interest (m).
        lat : `float`
            Latitude angle to point of interest (rad).
        lon : `float`
            Longitude to the point of interest (rad).
            
        Returns
        -------
        D : `float`
            The derivative of the harmonics potential wrt theta at the point.
            
        """
        GM = self.planet.GM
        a = self.planet.a
        D = 0.0

        nmLegendreCoeffs, cosines, sines = self._getLCS(lat,
                                                        lon,
                                                        nmLegendreCoeffs,
                                                        cosines,
                                                        sines)
        
        for k in range(2,self.Nharmonics+1):
            egmIndex = int((k-2)*(k+3)/2)
            mv=nmLegendreCoeffs[k,0:(k+1)]
            cs = sum([mv[l]*(-l*sines[l]*self.harmonics[egmIndex+l][2] 
                             + l*cosines[l]*self.harmonics[egmIndex+l][3]) 
                      for l in range(0,k+1)])
            D += GM*cs*(a/r)**k/r

        return D 
       
    def harmonicsdelTdelT(self, 
                      r, 
                      lat, 
                      lon,
                      nmLegendreCoeffs = None,
                      cosines = None,
                      sines = None):
        """
        Compute the second derivative of the harmonics potential with respect 
        to T (theta) twice.
        
        Parameters
        ----------
        r : `float`
            Range from centre of the earth to point of interest (m).
        lat : `float`
            Latitude angle to point of interest (rad).
        lon : `float`
            Longitude to the point of interest (rad).
            
        Returns
        -------
        D : `float`
            The second derivative of the harmonics potential wrt T twice at the 
            point of interest.
            
        """
        GM = self.planet.GM
        a = self.planet.a
        D = 0.0

        nmLegendreCoeffs, cosines, sines = self._getLCS(lat,
                                                        lon,
                                                        nmLegendreCoeffs,
                                                        cosines,
                                                        sines)
        
        for k in range(2,self.Nharmonics+1):
            egmIndex = int((k-2)*(k+3)/2)
            mv=nmLegendreCoeffs[k,0:(k+1)]
            cs = sum([mv[l]*(-l**2*cosines[l]*self.harmonics[egmIndex+l][2] 
                             - l**2*sines[l]*self.harmonics[egmIndex+l][3]) 
                      for l in range(0,k+1)])
            D += GM*cs*(a/r)**k/r

        return D 
    
    def harmonicsdelTdelR(self, 
                      r, 
                      lat, 
                      lon,
                      nmLegendreCoeffs = None,
                      cosines = None,
                      sines = None):
        """
        Compute the second derivative of the harmonics potential with respect
        to T and R.
        
        Parameters
        ----------
        r : `float`
            Range from centre of the earth to point of interest (m).
        lat : `float`
            Latitude angle to point of interest (rad).
        lon : `float`
            Longitude to the point of interest (rad).
            
        Returns
        -------
        D : `float`
            The second derivative of the harmonics potential wrt T and R at the 
            point of interest.
            
        """
        GM = self.planet.GM
        a = self.planet.a
        D = 0.0

        nmLegendreCoeffs, cosines, sines = self._getLCS(lat,
                                                        lon,
                                                        nmLegendreCoeffs,
                                                        cosines,
                                                        sines)
        
        for k in range(2,self.Nharmonics+1):
            egmIndex = int((k-2)*(k+3)/2)
            mv=nmLegendreCoeffs[k,0:(k+1)]
            cs = sum([mv[l]*(-l*sines[l]*self.harmonics[egmIndex+l][2] 
                             + l*cosines[l]*self.harmonics[egmIndex+l][3]) 
                      for l in range(0,k+1)])
            D += -(k+1.0)*GM*cs*(a/r)**k/r**2

        return D 

    # U corresponds to latitude
    def harmonicsdelU(self, 
                  r, 
                  lat, 
                  lon,
                  nmLegendreCoeffs = None,
                  cosines = None,
                  sines = None):
        """
        Compute the derivative of the harmonics potential with respect to U.
        
        This is the component of gravitational acceleration in the direction 
        of the unit vector in latitude. More or less acceleration in the north
        south direction.
        
        Parameters
        ----------
        r : `float`
            Range from centre of the earth to point of interest (m).
        lat : `float`
            Latitude angle to point of interest (rad).
        lon : `float`
            Longitude to the point of interest (rad).
            
        Returns
        -------
        D : `float`
            The derivative of the harmonics potential wrt U at the point.
            
        """
        GM = self.planet.GM
        a = self.planet.a
        u = sin(lat)
        v = cos(lat)
        D = 0.0

        nmLegendreCoeffs, cosines, sines = self._getLCS(lat,
                                                        lon,
                                                        nmLegendreCoeffs,
                                                        cosines,
                                                        sines)
        
        for k in range(2,self.Nharmonics+1):
            egmIndex = int((k-2)*(k+3)/2)
            mv1 = nmLegendreCoeffs[k,0:(k+1)]
            mv2 = nmLegendreCoeffs[(k-1),0:(k+1)]
            f1 = [mv1[l]*u*k for l in range(0, k+1)]
            f2 = [mv2[l]*sqrt((k**2-l**2)*(k+0.5)/(k-0.5)) 
                  for l in range(0, k+1)]
            
            cs = sum([(f1[l]-f2[l])*(cosines[l]*self.harmonics[egmIndex+l][2] 
                                     + sines[l]*self.harmonics[egmIndex+l][3]) 
                      for l in range(0,k+1)])
            D+= v*GM*cs*(a/r)**k/(u**2-1.0)/r

        return D 

    def fkl(self, k, l):
        if k>l:
            return np.sqrt((k**2-l**2)*(k+0.5)/(k-0.5))
        else:
            return 0.0
    
    def harmonicsdelUdelU(self, 
                      r, 
                      lat, 
                      lon,
                      nmLegendreCoeffs = None,
                      cosines = None,
                      sines = None):
        """
        Compute the second derivative of the harmonics potential with respect 
        to U twice.
        
        Parameters
        ----------
        r : `float`
            Range from centre of the earth to point of interest (m).
        lat : `float`
            Latitude angle to point of interest (rad).
        lon : `float`
            Longitude to the point of interest (rad).
            
        Returns
        -------
        D : `float`
            The second derivative of the harmonics potential wrt U twice at the 
            point of interest.
            
        """
        GM = self.planet.GM
        a = self.planet.a
        u = sin(lat)
        v = cos(lat)
        D = 0.0

        nmLegendreCoeffs, cosines, sines = self._getLCS(lat,
                                                        lon,
                                                        nmLegendreCoeffs,
                                                        cosines,
                                                        sines)
        
        for k in range(2,self.Nharmonics+1):
            egmIndex = int((k-2)*(k+3)/2)
            mv1 = nmLegendreCoeffs[k,0:(k+1)]
            mv2 = nmLegendreCoeffs[(k-1),0:(k+1)]
            mv3 = nmLegendreCoeffs[(k-2),0:(k+1)]
            f1 = [mv1[l]*(k**2*u**2-k)/v**2 for l in range(0, k+1)]
            f2 = [mv2[l]*2.0*(1.0-k)*u/v**2*self.fkl(k,l) 
                  for l in range(0, k+1)]
            f3 = [mv3[l]*self.fkl(k,l)*self.fkl(k-1,l)/v**2 
                  for l in range(0, k+1)]
            
            cs = sum([(f1[l]+f2[l]+f3[l])*(cosines[l]*self.harmonics[egmIndex+l][2] 
                                           + sines[l]*self.harmonics[egmIndex+l][3]) 
                      for l in range(0,k+1)])
            D += GM*cs*(a/r)**k/r

        return D
    
    def harmonicsdelUdelT(self, 
                      r, 
                      lat, 
                      lon,
                      nmLegendreCoeffs = None,
                      cosines = None,
                      sines = None):
        """
        Compute the second derivative of the harmonics potential with respect 
        to U and T.
        
        Parameters
        ----------
        r : `float`
            Range from centre of the earth to point of interest (m).
        lat : `float`
            Latitude angle to point of interest (rad).
        lon : `float`
            Longitude to the point of interest (rad).
            
        Returns
        -------
        D : `float`
            The second derivative of the harmonics potential wrt U and T at the 
            point of interest.
            
        """
        GM = self.planet.GM
        a = self.planet.a
        u = sin(lat)
        v = cos(lat)
        D = 0.0

        nmLegendreCoeffs, cosines, sines = self._getLCS(lat,
                                                        lon,
                                                        nmLegendreCoeffs,
                                                        cosines,
                                                        sines)
        
        for k in range(2,self.Nharmonics+1):
            egmIndex = int((k-2)*(k+3)/2)
            mv1=nmLegendreCoeffs[k,0:(k+1)]
            mv2 = nmLegendreCoeffs[(k-1),0:(k+1)]
            f1 = [mv1[l]*u*k for l in range(0, k+1)]
            f2 = [mv2[l]*sqrt((k**2-l**2)*(k+0.5)/(k-0.5)) 
                  for l in range(0, k+1)]
            
            cs = sum([(f1[l]-f2[l])*(-l*sines[l]*self.harmonics[egmIndex+l][2] 
                                     + l*cosines[l]*self.harmonics[egmIndex+l][3]) 
                      for l in range(0,k+1)])
            D+= GM*v*cs*(a/r)**k/(u**2-1.0)/r

        return D 

    def harmonicsdelUdelR(self, 
                      r, 
                      lat, 
                      lon,
                      nmLegendreCoeffs = None,
                      cosines = None,
                      sines = None):
        """
        Compute the second derivative of the harmonics potential with respect 
        to U and R.
        
        Parameters
        ----------
        r : `float`
            Range from centre of the earth to point of interest (m).
        lat : `float`
            Latitude angle to point of interest (rad).
        lon : `float`
            Longitude to the point of interest (rad).
            
        Returns
        -------
        D : `float`
            The second derivative of the harmonics potential wrt U and R at the 
            point of interest.
            
            
        """
        GM = self.planet.GM
        a = self.planet.a
        u = sin(lat)
        v = cos(lat)
        D = 0.0

        nmLegendreCoeffs, cosines, sines = self._getLCS(lat,
                                                        lon,
                                                        nmLegendreCoeffs,
                                                        cosines,
                                                        sines)
        
        for k in range(2,self.Nharmonics+1):
            egmIndex = int((k-2)*(k+3)/2)
            mv1=nmLegendreCoeffs[k,0:(k+1)]
            mv2 = nmLegendreCoeffs[(k-1),0:(k+1)]
            f1 = [mv1[l]*u*k for l in range(0, k+1)]
            f2 = [mv2[l]*sqrt((k**2-l**2)*(k+0.5)/(k-0.5)) 
                  for l in range(0, k+1)]
            
            cs = sum([(f1[l]-f2[l])*(cosines[l]*self.harmonics[egmIndex+l][2] 
                                     + sines[l]*self.harmonics[egmIndex+l][3]) 
                      for l in range(0,k+1)])
            D+= -(k+1.0)*v*GM*cs*(a/r)**k/(u**2-1.0)/r**2

        return D 

    def Jsx(self, X):
        """
        Compute the derivatives of ECEF cartesian coordinates with respect to
        spherical polar coordinates.
        
        This is the Jacobian between XYZ and RTU
        
        Parameters
        ----------
        X : `numpy.ndarray`, (3,)
            Cartesian coordinates.
            
        Returns
        -------
        `numpy.ndarray`, (3,3)
            The Jacobian matrix.
            
        """
        r = np.linalg.norm(X[0:3])
        p = np.sqrt(X[0]**2 + X[1]**2)
        
        return np.array([X/r, 
                         [-X[0]*X[2]/r**2/p, -X[1]*X[2]/r**2/p, p/r**2], 
                         [-X[1]/p**2, X[0]/p**2, 0.0]
                        ])
    
    def secondDtve(self, X):
        """
        Compute the a re-ordered form of the Hessian matrix between XYZ and 
        RTU
        
        See equation B.24
        
        Parameters
        ----------
        X : `numpy.ndarray`, (3,)
            The Cartesian coordinates.
            
        Returns
        -------
        `numpy.ndarray`, (9,9)
            The reordered Hessian matrix.
            
        """
        x = X[0]
        y = X[1]
        z = X[2]
        
        r = np.linalg.norm(X[0:3])
        p = np.sqrt(x**2 + y**2)
        
        Jtr = 1.0/r*np.eye(3) - 1.0/r**3*np.outer(X[0:3],X[0:3])
    
        Jtu = np.array([[-z*((r*y)**2-2.0*(x*p)**2), 
                         x*y*z*(2.0*p**2+r**2), 
                         -x*(p**4-(p*z)**2)],
                        [x*y*z*(2.0*p**2+r**2), 
                         -z*((r*x)**2-2.0*(y*p)**2), 
                         -y*(p**4-(z*p)**2)],
                        [-x*(p**4-(p*z)**2), 
                         -y*(p**4-(z*p)**2), 
                         -2.0*z*p**4]])/p**3/r**4
    
        Jtt = np.array([[2.0*x*y, y**2-x**2, 0.0],
                        [y**2-x**2, -2.0*x*y, 0.0],
                        [0.0,0.0,0.0]])/p**4
        
        reshuffle = np.array([[1,0,0,0,0,0,0,0,0],
                              [0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0],
                              [0,1,0,0,0,0,0,0,0],
                              [0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,0,1,0],
                              [0,0,1,0,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,0,1]])
        #return reshuffle.dot(np.array(list(Jtr)+list(Jtu)+list(Jtt)))
        return reshuffle.dot(np.concatenate((Jtr, Jtu, Jtt)))
    
        
    # def satEQMold(self,X,t):
    #     """
    #     Return the satellite equations of motion for the given state vector
    #     X
        
    #     Return M from the first-order system of differential equations such 
    #     that dX/dt = MX
        
    #     Parameters
    #     ----------
    #     X : `numpy.ndarray`, (6,)
    #         Six element array describing the state vector.
    #     t : `float`
    #         The time of the state vector information in s.
            
    #     Returns
    #     -------
    #     `numpy.ndarray`, (6,)
    #         Satellite velocity and acceleration.
            
    #     """
    #     # Define some constants
    #     wE = self.planet.w
        
    #     # Get n-body positions
    #     nbody = self.planet.nbodyacc(t, X[0:3])
        
    #     # Transform to lat/long/r spherical polar
    #     llh = self.xyz2SphericalPolar(X)
    #     lat, lon = np.radians(llh[0:2])

    #     # Compute the norm
    #     r = llh[2]
    #     p = np.sqrt(X[0]**2 + X[1]**2)
    #     Xp = mat(X[0:3])
    #     Xv = mat(X[3:])

    #     nmLegendreCoeffs, cosines, sines = self._getLCS(lat, lon)
        
        
    #     # nmLegendreCoeffs = self.myLegendre(self.Nharmonics, sin(lat))
    #     # cosines = [cos(l*lon) for l in range (0, self.Nharmonics+1)]
    #     # sines = [sin(l*lon) for l in range (0, self.Nharmonics+1)]
    #     # Compute the component of the gradient due to r, theta, phi
    #     delVdelR, delVdelU, delVdelT = self.harmonicsdel(r,
    #                                                   lat,
    #                                                   lon,
    #                                                   nmLegendreCoeffs,
    #                                                   cosines = cosines,
    #                                                   sines = sines)

    #     # Calculate factors to transform sperical polar to Cartesian 
    #     # derivatives
    #     delVdelRX = delVdelR/r*mat(X[0:3])
    #     delVdelUX = delVdelU*mat([-X[0]*X[2]/r**2/p, 
    #                               -X[1]*X[2]/r**2/p, 
    #                               p/r**2])
    #     delVdelTX = delVdelT*mat([-X[1]/p**2, X[0]/p**2, 0.0])


    #     # Compute some shifting matrices
    #     I2 = mat([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    #     Q2 = mat([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        
    #     XM = mat([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    #               [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    #     VM = mat([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    #               [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    #               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        
        
    #     # Do the calculation
    #     return np.array(X[3:]*XM + (wE**2*Xp*I2 
    #                                 + 2.0*wE*Xv*Q2 
    #                                 + delVdelRX 
    #                                 + delVdelTX 
    #                                 + delVdelUX)*VM).flatten()
    
            
    def satEQM(self,X,t):
        """
        Return the satellite equations of motion for the given state vector
        X
        
        Return M from the first-order system of differential equations such 
        that dX/dt = MX
        
        Parameters
        ----------
        X : `numpy.ndarray`, (6,)
            Six element array describing the state vector.
        t : `float`
            The time of the state vector information in s.
            
        Returns
        -------
        `numpy.ndarray`, (6,)
            Satellite velcoity and acceleration.
            
        """
        # Define some constants
        wE = self.planet.w
        
        # Get n-body positions
        nbody = self.planet.nbodyacc(t, X)
        drag = self.planet.dragacc(X)
        
        # Transform to lat/long/r spherical polar
        llh = self.xyz2SphericalPolar(X)
        lat, lon = np.radians(llh[0:2])

        # Compute the norm
        r = llh[2]
        p = np.sqrt(X[0]**2 + X[1]**2)
        Xp = X[0:3]
        Xv = X[3:]

        nmLegendreCoeffs, cosines, sines = self._getLCS(lat, lon)
        
        
        # nmLegendreCoeffs = self.myLegendre(self.Nharmonics, sin(lat))
        # cosines = [cos(l*lon) for l in range (0, self.Nharmonics+1)]
        # sines = [sin(l*lon) for l in range (0, self.Nharmonics+1)]
        # Compute the component of the gradient due to r, theta, phi
        delVdelR, delVdelU, delVdelT = self.harmonicsdel(r,
                                                     lat,
                                                     lon,
                                                     nmLegendreCoeffs,
                                                     cosines = cosines,
                                                     sines = sines)

        # Calculate factors to transform sperical polar to Cartesian 
        # derivatives
        delVdelRX = delVdelR/r*X[0:3]
        delVdelUX = delVdelU*np.array([-X[0]*X[2]/r**2/p, 
                                       -X[1]*X[2]/r**2/p, 
                                       p/r**2])
        delVdelTX = delVdelT*np.array([-X[1]/p**2, 
                                       X[0]/p**2, 
                                       0.0])


        # Compute some shifting matrices
        I2 = np.array([[1.0, 0.0, 0.0], 
                       [0.0, 1.0, 0.0], 
                       [0.0, 0.0, 0.0]])
        Q2 = np.array([[0.0, 1.0, 0.0], 
                       [-1.0, 0.0, 0.0], 
                       [0.0, 0.0, 0.0]])
        
        XM = np.concatenate((np.eye(3), np.zeros((3,3))))
        VM = np.concatenate((np.zeros((3,3)), np.eye(3)))
        
        
        # Do the calculation
        return XM.dot(Xv) + VM.dot(wE**2*I2.dot(Xp) 
                                   + 2.0*wE*Q2.dot(Xv) 
                                   + delVdelRX 
                                   + delVdelTX 
                                   + delVdelUX
                                   + nbody
                                   + drag)
    
    
    def isatEQM(self, t, X):
        """
        Function to compute the satellite equations of motion
        
        This function computes the satellite equations of motion as done
        in :func:`~measurement.satEQM`, but with the parameters reversed.
        This helps with calling the scipy.ivp differential equation
        solver.
        
        Parameters
        ----------
        t : `float`
            The time of the state vector information in s.
        X : `numpy.ndarray`, (6,)
            Six element array describing the state vector.
            
        Returns
        -------
        `numpy.ndarray`, (6,)
            The values for the equations of motion (velocity and acceleration)
            at the point in time.
            
        """
        return self.satEQM(X,t)
    
    def secondSphericalDerivative(self, r, lat, lon):
        """
        Calculate a matrix of second dervatives of the harmonics potential
        
        See the second part of equation B.24. This is needed to find the
        jerk
        
        Parameters
        ----------
        r : `float`
            Range from centre of the earth to point of interest (m).
        lat : `float`
            Latitude angle to point of interest (rad).
        lon : `float`
            Longitude to the point of interest (rad).
            
        Returns
        -------
        `numpy.ndarray`, (3,3)
            Matrix of second derivatives of the harmonics potential.
            
        """
        nmLegendreCoeffs, cosines, sines = self._getLCS(lat, lon)
        
        h11, h12, h13, h22, h23, h33 = self.harmonicsdeldel(r, 
                                                        lat, 
                                                        lon,
                                                        nmLegendreCoeffs = nmLegendreCoeffs,
                                                        cosines = cosines,
                                                        sines = sines)

        return np.array([[h11, h12, h13],
                         [h12, h22, h23],
                         [h13, h23, h33]])
        
    def expandedState(self,X, t=0):
        """
        Compute the expanded state vector
        
        Compute the acceleration and jerk components of the state vector. The
        input is the position and velocity. These are used to compute the 
        acceleration and jerk. The output is the state vector appended by the
        acceleration and jerk
        
        Parameters
        ----------
        X : `numpy.ndarray`, (6,)
            Six element state vector.
        t : `float`
            Time associate with the state vector.
            
        Returns
        -------
        `numpy.ndarray`, (4,3)
            A 4x3 array with the top row ecef_X, second row ecef_dX/dt, third
            row ecef_d2X/dt2, fourth row ecef_d3X/dt3.
            
        """
        # Define some constants
        wE = self.planet.w
        
        # Transform to lat/long/r spherical polar
        llh = self.xyz2SphericalPolar(X)
        lat, lon = np.radians(llh[0:2])

        # Compute the norm
        r = llh[2]
        ecef_X = X[:3]
        ecef_dX = X[3:]
        
        # Compute the component of the gradient due to r, theta, phi
        nmLegendreCoeffs, cosines, sines = self._getLCS(lat, lon)
        
        d0, d1, d2 = self.harmonicsdel(r, 
                                       lat, 
                                       lon,
                                       nmLegendreCoeffs = nmLegendreCoeffs,
                                       cosines = cosines,
                                       sines = sines)
        
        inertial_ddS = np.array([d0, d1, d2])
        
        # Compute an expanded ddS
        exp_ddS = np.array([[d0, d1, d2, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, d0, d1, d2, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, d0, d1, d2]])
        
        #print("Second derivative w.r.t. rxixj")
        # start = timer()
        dtwosdx = self.secondDtve(X)
        # end = timer()
        # print("2: %0.4f" % (end-start))
    
        # Calculate the U Hessian matrix
        # start = timer()
        Hu = self.secondSphericalDerivative(r, lat, lon)
        # end = timer()
        # print("3: %0.4f" % (end-start))


        # Compute some shifting matrices
        I2 = np.array([[1.0, 0.0, 0.0], 
                       [0.0, 1.0, 0.0], 
                       [0.0, 0.0, 0.0]])
        Q2 = np.array([[0.0, 1.0, 0.0], 
                       [-1.0, 0.0, 0.0], 
                       [0.0, 0.0, 0.0]])        
        
        # Calculate factors to transform sperical polar to Cartesian 
        # derivatives
        # start = timer()
        J = self.Jsx(ecef_X)
        # end = timer()
        # print("4: %0.4f" % (end-start))
        JT = J.T
        
        # Print the second derivatives
        F = exp_ddS.dot(dtwosdx) + JT.dot(Hu.dot(J))
        inertial_dX = ecef_dX - wE*np.dot(Q2, ecef_X)
        inertial_ddX = np.dot(JT, inertial_ddS)
        inertial_dddX = F.dot(inertial_dX)

        # Calculate te seoncd derivative in ECEF
        ecef_ddX = (wE**2*np.dot(I2,ecef_X) 
                    + 2.0*wE*np.dot(Q2, ecef_dX) 
                    + inertial_ddX)
        
        
        # Calculate the third derivative in ECEF
        ecef_dddX = (2.0*wE**3*np.dot(Q2, ecef_X) 
             - 3.0*wE**2*np.dot(I2, ecef_dX) 
             + 3.0*wE*np.dot(Q2, inertial_ddX) 
             + inertial_dddX)
        
        
        return np.array([ecef_X,
                         ecef_dX,
                         ecef_ddX,
                         ecef_dddX])
    
    
    def myLegendre_old(self, n0, t):
        n0 += 1
        n = max(n0, 2)
        
        [M, N] = meshgrid(arange(float(n)), arange(float(n)))
        
        # Select only valid indeces
        idx = N > M
        
        # Define the A matrix
        A = zeros([n0, n0])
        A[idx] = sqrt((2*N[idx]-1)*(2*N[idx]+1)/(N[idx]-M[idx])/(N[idx]+M[idx]))
        
        # Define the B matrix
        B = zeros([n0, n0])
        B[idx] = sqrt((2*N[idx]+1)*(N[idx]+M[idx]-1)*(N[idx]-M[idx]-1)
                      /((N[idx]-M[idx])*(N[idx]+M[idx])*(2*N[idx]-3)))

        # Define the diagonal p matrix (the m,m terms)
        u = sqrt(1.0-t**2)
        ff = [1.0, u] + [u*sqrt((2*i+1)/(2*i)) for i in arange(2.0,float(n))]
        gg = cumprod(ff)*sqrt(3.0)
        gg[0] = 1.0
        p = diag(gg)
        p[1,:] += t*A[1,:]*p[0,:]
        
        for k in range(2,n):
            p[k,:] += t*A[k,:]*p[k-1,:] - B[k,:]*p[k-2,:]
            
        if(n0<2):
            return p[0:n0, 0:n0]
            
        return p
    
    def myLegendre(self, n0, t):
        """
        Compute the fully normalized associated Legendre polynomials up to 
        degree n for argument x.
        
        See `normalized lpnm <http://mitgcm.org/~mlosch/geoidcookbook/node11.html>`_ 
        for a description of the fully normalized associated Legendre 
        polynomials
        
        Parameters
        ----------
        n : `int`
            The maximum degree of the polynomials.
        x : `float`
            The argument of the function.
            
        Returns
        -------
        `numpy.ndarray`, (n+1, n+1)
            (n+1)x(n+1) array with the evaluated fully normalized associated
            Legendre polynomials of degree 0 to n along with order 0 to n
            
        Notes
        -----
        Reproduced from method of Holmes and Featherstone (2002)::
            Holmes, S. A. and W. E. Featherstone, 2002. A unified approach to 
            the Clenshaw summation and the recursive computation of very high 
            degree and order normalised associated Legendre functions Journal 
            of Geodesy, 76(5), pp. 279-299.
            
        """
        p = np.zeros((n0+1, n0+1))
        myLegendre_numba(p, n0, t)
        return p

    def toInertial(self,mData, t):
        return self.toPCI(mData, t)
        # w = self.planet.w;
        # X = mat([[cos(w*t),-sin(w*t),0.0],
        #          [sin(w*t),cos(w*t),0.0],
        #          [0.0,0.0,1.0]])
        # V = w*mat([[-sin(w*t),-cos(w*t),0.0],
        #            [cos(w*t),-sin(w*t),0.0],
        #            [0.0,0.0,0.0]])
        # ix = X*mat(mData[0:3]).H
        # iv = X*mat(mData[3:6]).H + V*mat(mData[0:3]).H
        # ivect = []
        # for pos in ix:
        #     ivect.append(float(pos))
        # for vel in iv:
        #     ivect.append(float(vel))
        # return ivect
    
    def toPCI(self, mData, t, w0 = 0.0):
        w = self.planet.w;
        cosWT = np.cos(w*t + w0)
        sinWT = np.sin(w*t + w0)
        X = np.array([[cosWT, -sinWT, 0],
                      [sinWT,  cosWT, 0],
                      [0,      0,     1]])
        V = w*np.array([[-sinWT, -cosWT, 0],
                        [cosWT,  -sinWT, 0],
                        [0,      0,      0]])

        ix = X.dot(mData[0:3])
        iv = X.dot(mData[3:6]) + V.dot(mData[0:3])
        return np.concatenate((ix, iv))
    
    def toECEF(self,mData, t):
        return self.toPCR(mData, t)
        # w = self.planet.w;
        # X = mat([[cos(w*t),sin(w*t),0.0],
        #          [-sin(w*t),cos(w*t),0.0],
        #          [0.0,0.0,1.0]])
        # V = w*mat([[-sin(w*t),cos(w*t),0.0],
        #            [-cos(w*t),-sin(w*t),0.0],
        #            [0.0,0.0,0.0]])
        # ix = X*mat(mData[0:3]).H
        # iv = X*mat(mData[3:6]).H + V*mat(mData[0:3]).H
        # ivect = []
        # for k in range(0,3):
        #     ivect.append(float(ix[k]))
        # for k in range(0,3):
        #     ivect.append(float(iv[k]))
        # return ivect
    
    def toPCR(self, mData, t, w0 = 0.0):
        w = self.planet.w;
        cosWT = np.cos(w*t + w0)
        sinWT = np.sin(w*t + w0)
        X = np.array([[cosWT,  sinWT, 0],
                      [-sinWT, cosWT, 0],
                      [0,      0,     1]])
        V = w*np.array([[-sinWT,  cosWT, 0],
                        [-cosWT, -sinWT, 0],
                        [0,      0,     0]])

        ix = X.dot(mData[0:3])
        iv = X.dot(mData[3:6]) + V.dot(mData[0:3])
        return np.concatenate((ix, iv))

    def integrate(self, 
                  t, 
                  k, 
                  t_eval=None, 
                  rtol = 3.0e-14, 
                  atol = 1.0e-14,
                  intmethod = 'RK45'):
        """
        Numerically integrate satEQM
        
        This function uses the
        `scipy odeint <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html>`_
        function
        
        Parameters
        ----------
        t : `float`
            Integration time (s).
        k : `int`
            Index into list of state vectors so we know from where we are
            integrating.
            
        Returns
        -------
        istate : `numpy.ndarray`, (6,)
            The value of the numerical integration as a state vector.
            
        """
        self.reference_time = self.measurementTime[k]
        
        if t_eval is None:
            y = solve_ivp(self.isatEQM, 
                          t, 
                          self.measurementData[k], 
                          method=intmethod, 
                          rtol=rtol,
                          atol=atol)
        else:
            y = solve_ivp(self.isatEQM, 
                          t, 
                          self.measurementData[k], 
                          method=intmethod, 
                          t_eval=t_eval, 
                          rtol=rtol, 
                          atol=atol)
        return y
        
    def estimateTimeRange(self, 
                          dtime, 
                          integrationTimes = None,
                          rtol = 3e-14,
                          atol = 1e-14,
                          intmethod = 'RK45'):
        """
        Integrate the satEQM to estimate the state vectors over a range of 
        times.
        
        This function uses the integrate function to compute the values
        of the state vector at different times.
        
        Parameters
        ----------
        dtime : `list`, [`datetime.datetime`]
            A list of times at which to compute/estimate the state vectors. 
            This is given as a list of datetimes.
            
        Returns
        -------
        `list`, [`numpy.ndarray`, (6,)]
            The computed/estimated state vectors.
            
        """
        if type(dtime[0]) == datetime.datetime:
            mnTime = sum([(tm - dtime[0]).total_seconds() for tm in dtime])/len(dtime)
            minK = self.findNearest(dtime[0] + datetime.timedelta(seconds=mnTime))
        else:
            mnTime = sum([(tm - dtime[0])/np.timedelta64(1,'s') for tm in dtime])/len(dtime)
            minK = self.findNearest(dtime[0] + np.timedelta64(int(1e9*mnTime), 'ns'))
            
        #print(self.measurementTime[minK])
        if integrationTimes is None:
            if type(self.measurementTime[0]) == datetime.datetime:
                integrationTimes = [(dt - self.measurementTime[minK]).total_seconds() 
                                    for dt in dtime]
            else:
                integrationTimes = [(dt - self.measurementTime[minK])/np.timedelta64(1,'s') 
                                    for dt in dtime]
                
        
        # Perform the integration
        # Get integration times
        pTimes = [0.0] + [tm for tm in integrationTimes if tm > 0]
        nTimes = [tm for tm in integrationTimes if tm < 0] + [0.0]
        nTimes.reverse()
           
        if(len(nTimes) > 1):
            nState = self.integrate([np.max(nTimes), np.min(nTimes)], 
                                    minK, 
                                    t_eval = nTimes,
                                    rtol = rtol,
                                    atol = atol,
                                    intmethod = intmethod)
        else:
            nState = SimpleNamespace(y=np.ones((6,0)))
            
        # Integrate
        if(len(pTimes) > 1):
            pState = self.integrate([np.min(pTimes), np.max(pTimes)], 
                                    minK, 
                                    t_eval = pTimes,
                                    rtol = rtol,
                                    atol = atol,
                                    intmethod = intmethod)
        else:
            pState = SimpleNamespace(y=np.ones((6,0)))

            
        # Create and return the output
        if 0.0 in integrationTimes:
            y_array = np.concatenate((np.fliplr(nState.y[:,1:]), pState.y), 
                                     axis=1).T
        else:
            y_array = np.concatenate((np.fliplr(nState.y[:,1:]), 
                                      pState.y[:,1:]), axis=1).T

            
        return y_array
    
    def estimate(self, 
                 dtime,
                 rtol = 3e-14,
                 atol = 1e-14,
                 intmethod = 'RK45'):
        """
        Estimate/Compute the state vector at a particular time
        
        This method uses the class integrate function to numerically
        integrate the ODE from the closest state vector to the desired 
        time point
        
        Parameters
        ----------
        dtime : `datetime.datetime`
            The desired time at which to estimate the state vector.
            
        Returns
        -------
        `numpy.ndarray`, (6,)
            The computed/estimated state vector.
            
        """
        minK = self.findNearest(dtime)
        self.reference_time = dtime
        if type(self.measurementTime[0]) == datetime.datetime:
            dT = [0.0, (dtime - self.measurementTime[minK]).total_seconds()]
        else:
            dT = [0.0, (dtime 
                        - self.measurementTime[minK])/np.timedelta64(1,'s')]
        
        # Do the integration
        sol = self.integrate(dT,
                             minK,
                             rtol = rtol,
                             atol = atol,
                             intmethod = intmethod)
        
        if not dtime in self.measurementTime:
            self.add(dtime, sol.y[:,-1])
        return sol.y[:,-1]
        
class state_vector_TSX(state_vector):
    """
    Class to load TSX format state vectors
    
    Methods
    -------
    readStateVectors:
        Reads state vectors from a TSX style XML file
        
    """
    def readStateVectors(self, XMLfile):
        # this function will read TSX XML state 
        # vectors
        xmlroot = etree.parse(XMLfile).getroot()
        stateVectorList = xmlroot.findall('.//stateVec')
        
        for sv in stateVectorList:
            svTime = sv.find('timeUTC')
            x = sv.find('posX')
            y = sv.find('posY')
            z = sv.find('posZ')
            vx = sv.find('velX')
            vy = sv.find('velY')
            vz = sv.find('velZ')
            self.add(datetime.datetime.strptime(svTime.text, 
                                                '%Y-%m-%dT%H:%M:%S.%f'), 
              [float(x.text),
               float(y.text),
               float(z.text),
               float(vx.text),
               float(vy.text),
               float(vz.text)])
        return

class state_vector_Radarsat(state_vector):
    """
    Class to load Radarsat-2 format state vectors
    
    Methods
    -------
    readStateVectors:
        Reads state vectors from a Radarsat-2 style XML file
        
    """
    def readStateVectors(self, XMLFile):
        # this function will read Radarsat2 product.xml state 
        # vectors
#        parser = etree.XMLParser(remove_blank_text=True)
        dataPool = etree.parse(XMLFile).getroot()
        stateVectorList = dataPool.find('.//stateVectorList')
        stateVectorTimeElements = stateVectorList.findall('stateVector/time')
        svX = stateVectorList.findall('stateVector/xPos')
        svY = stateVectorList.findall('stateVector/yPos')
        svZ = stateVectorList.findall('stateVector/zPos')
        svVX = stateVectorList.findall('stateVector/xVel')
        svVY = stateVectorList.findall('stateVector/yVel')
        svVZ = stateVectorList.findall('stateVector/zVel')
        
        for sv,x,y,z,vx,vy,vz in zip(stateVectorTimeElements,
                                     svX,
                                     svY,
                                     svZ,
                                     svVX,
                                     svVY,
                                     svVZ):
            self.add(self.getDateTimeXML(sv), [float(x.text),
                                               float(y.text),
                                               float(z.text),
                                               float(vx.text),
                                               float(vy.text),
                                               float(vz.text)])
        return
        
class state_vector_RSO(state_vector):
    """
    Class to load RSO format state vectors
    
    Methods
    -------
    readStateVectors:
        Reads state vectors from an RSO style text file. See
        `here isdc-old.gfz-potsdam.de`
        
    """
    def readStateVectors(self, RSOFile):
        # This function will read a CHAMP format RSO file
        # See site isdc-old.gfz-potsdam.de
        TTUTC_offset = 0.0
        refDate = datetime.datetime(2000, 1, 1, 0, 0, 0, 0)
        
        def ingestLine(ln):
            # Split the line according to the CHAMP format
            if(len(ln) < 90):
                return
            day = float(ln[0:6])*1e-1
            secsSinceMidnight = float(ln[6:17])*1e-6 - TTUTC_offset
            self.add(refDate + datetime.timedelta(days=math.ceil(day), 
                                                  seconds = secsSinceMidnight),
                [float(ln[17:29])*1e-3,
                float(ln[29:41])*1e-3,
                float(ln[41:53])*1e-3,
                float(ln[53:65])*1e-7,
                float(ln[65:77])*1e-7,
                float(ln[77:89])*1e-7])
            return
            
            
        # Open the file
        with open(RSOFile, 'r') as rso:
            readOrbit = False
            for line in rso:
                if(readOrbit):
                    # Add the data here
                    ingestLine(line)
                dummy = line.split()
                if('TT-UTC' in dummy):
                    TTUTC_offset = float(dummy[1])*1e-3
                if('ORBIT' in dummy):
                    readOrbit = True
            
        
    
        
class state_vector_ESAEOD(state_vector):
    """
    Class to load ESA EOD format state vectors
    
    Methods
    -------
    readStateVectors:
        Reads state vectors from an ESA EOD style text file. See
        `Sentinel poeorb <https://qc.sentinel1.eo.esa.int/aux_poeorb>`_
        
    """
    def readStateVectors(self, 
                         EODFile, 
                         desiredStartTime=None, 
                         desiredStopTime=None):
        xmlroot = etree.parse(EODFile).getroot()
        osvlist = xmlroot.findall(".//OSV")

        for osv in osvlist:
            utcElement = osv.find(".//UTC")
            datetime.datetime
            utc = datetime.datetime.strptime(utcElement.text.split('=')[1],
                                             '%Y-%m-%dT%H:%M:%S.%f')
            if desiredStartTime is not None and desiredStopTime is not None:
                if(utc > desiredStartTime and utc < desiredStopTime):
                    # Add to list of state vectors
                    self.add(utc,
                                [float(osv.find(".//X").text), 
                                 float(osv.find(".//Y").text), 
                                 float(osv.find(".//Z").text), 
                                 float(osv.find(".//VX").text), 
                                 float(osv.find(".//VY").text), 
                                 float(osv.find(".//VZ").text)])
            else:
                # Add to list of state vectors
                self.add(utc,
                            [float(osv.find(".//X").text), 
                             float(osv.find(".//Y").text), 
                             float(osv.find(".//Z").text), 
                             float(osv.find(".//VX").text), 
                             float(osv.find(".//VY").text), 
                             float(osv.find(".//VZ").text)])
                
        
        return
        
class state_vector_Sarscape(state_vector):
    """
    Class to load Sarscape format state vectors
    
    Methods
    -------
    readStateVectors:
        Reads state vectors from an Sarscape style XML file.
        
    """
    def readStateVectors(self, SMLFile):
        dataPool = etree.parse(SMLFile).getroot()
        svX = dataPool.findall('.//{http://www.sarmap.ch/xml/SARscapeHeaderSchema}pos_x')
        svY = dataPool.findall('.//{http://www.sarmap.ch/xml/SARscapeHeaderSchema}pos_y')
        svZ = dataPool.findall('.//{http://www.sarmap.ch/xml/SARscapeHeaderSchema}pos_z')
        svVX = dataPool.findall('.//{http://www.sarmap.ch/xml/SARscapeHeaderSchema}vel_x')
        svVY = dataPool.findall('.//{http://www.sarmap.ch/xml/SARscapeHeaderSchema}vel_y')
        svVZ = dataPool.findall('.//{http://www.sarmap.ch/xml/SARscapeHeaderSchema}vel_z')

        #Find the time elements
        svN = dataPool.find('.//{http://www.sarmap.ch/xml/SARscapeHeaderSchema}StateVectorData')
        startTime = svN.find('.//{http://www.sarmap.ch/xml/SARscapeHeaderSchema}TimeFirst')
        timeDelta = svN.find('.//{http://www.sarmap.ch/xml/SARscapeHeaderSchema}TimeDelta')

        sTime = datetime.datetime.strptime(startTime.text[0:27], 
                                           '%d-%b-%Y %H:%M:%S.%f')
        tD = float(timeDelta.text)
        sTimes = [sTime+datetime.timedelta(seconds=tD*r) 
                  for r in range(len(svX))]

        # Populate the local data
        for sv,x,y,z,vx,vy,vz in zip(sTimes,svX,svY,svZ,svVX,svVY,svVZ):
            self.add(sv, [float(x.text),
                          float(y.text),
                          float(z.text),
                          float(vx.text),
                          float(vy.text),
                          float(vz.text)])
        return

class state_vector_Dimap(state_vector):
    """
    Class to load Dimap format state vectors
    
    Methods
    -------
    readStateVectors:
        Reads state vectors from a Dimap style XML file. This type of data
        are produced, for example, by ESA SNAP
        
    """
    def readStateVectors(self, dim):
        dataPool = etree.parse(dim).getroot()
        orbit = dataPool.find(".//MDElem[@name='Orbit_State_Vectors']")
        svX = orbit.findall(".//MDATTR[@name='x_pos']")
        svY = orbit.findall(".//MDATTR[@name='y_pos']")
        svZ = orbit.findall(".//MDATTR[@name='z_pos']")
        svVX = orbit.findall(".//MDATTR[@name='x_vel']")
        svVY = orbit.findall(".//MDATTR[@name='y_vel']")
        svVZ = orbit.findall(".//MDATTR[@name='z_vel']")
        sTimeText = orbit.findall(".//MDATTR[@name='time']")
        dformat = "%d-%b-%Y %H:%M:%S.%f"
        sTimes = [datetime.datetime.strptime(d.text, dformat) 
                  for d in sTimeText]

        # Populate the local data
        for sv,x,y,z,vx,vy,vz in zip(sTimes,svX,svY,svZ,svVX,svVY,svVZ):
            self.add(sv, [float(x.text),
                          float(y.text),
                          float(z.text),
                          float(vx.text),
                          float(vy.text),
                          float(vz.text)])
        return


@njit
def myLegendre_numba(p, n0, t):
    #n0 += 1
    N = max(n0+1, 2)
    
    # Define the diagonal p matrix (the m,m terms)
    u = sqrt(1.0-t**2)
    ff = np.array([1.0, u] + [u*sqrt((2*i+1)/(2*i)) 
                              for i in arange(2.0,float(N))])
    gg = cumprod(ff)*sqrt(3.0)
    gg[0] = 1.0
    
    # Define the A matrix
    A = np.zeros((N, N))
    B = np.zeros((N, N))
    for m in prange(N):
        p[m,m] = gg[m]
        for n in range(m):
            A[m,n] = sqrt((2*m-1)*(2*m+1)/(m-n)/(m+n))
            B[m,n] = sqrt((2*m+1)*(m+n-1)*(m-n-1)/
                          ((m-n)*(m+n)*(2*m-3)))

    #p = diag(gg)
    p[1,:] += t*A[1,:]*p[0,:]
    
    for k in range(2,N):
        p[k,:] += t*A[k,:]*p[k-1,:] - B[k,:]*p[k-2,:]
