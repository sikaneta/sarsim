import numpy as np

#from DEM.DEM import DEM
from space.planets import earth
import datetime


class satGeometry:
    """
    Class to handle geometry of SAR data
    
    Class to handle geometrical operations of SAR data
    
    Methods
    -------
    computeECEF
        Computes the ECEF coordinates of a point collected in the direction
        u from a given state vector at a given range at some height above the
        ellipsoid.
    computeLLHwithDEM
        Compute the lat/long/height as above, but using a DEM instead of a
        given HAE. Also outputs the lat/long/height rather than XYZ in 
        ECEF Cartesian coordinates.
    sysEq
        The system of equations for solving for XYZ on earth given the
        satellite statevector, the range to the target the height of the
        target above the ellipsoid and the look direction. See the following
        for a detailed `Geo Model PDF <_static/geo.pdf>`_
    sysJac
        The Jacobian of the system equations given by sysEq. See
        `Geo Model PDF`_
    computeECEF
        Python version of the methods in `Geo Model PDF`_ 
        to find the XYZ coordinate of a point on the ground given radar 
        imaging time, range, HAE and look angle
    computeLLHwithDEM
        Same as computeECEF, but instead of a given height, a DEM is used
        to calculate the height. If the height is given by the DEM in HAG,
        a function to convert HAG to HAE may be supplied
    geoF
        Forward functions to calculate Cartesian coordinates XYZ in ECEF from
        llh
    geoJ
        Jacobian of geoF. Used for Newton-Raphson inversion
    xyz2polar
        Calculate the polar coordinates (EPSG:4326) of a point given in ECEF
        Cartesian coordinates XYZ. This function uses Newton-Raphson to invert
        the forward transform given by geoF
    generateGeoGrid
        Generate a grid of points on a grid. This grid is given in radar space
        and converted to geo space
    createENVIGCP
        Write a set of GCPs to ENVI format for GCPs
    writeGCPtoSHP
        Write a set of GCPs to a shapefile
    writeGCPtoXML
        Write a set of GCPs to a vanilla XML file
    readGCPFile
        Read GCPs from a vanilla XML file
    geocodeSARNew
        Geo-reference a SAR file by using computed or to be computed GCPs. The
        gdal engine is used to do the actual warping
        
    """
    orbitFile = u"r:/SARorbits.xml"
    
    def __init__(self, body = earth()):
        self.body = body
        self.eS = np.array([body.a, body.a, body.b])

    # Compute the vector of functions
    def sysEq(self, X, range, u, h, sX, sV):
        """
        System of equations to relate radar coordinates to points on the 
        ground
        
        Returns the output of a system of equations relating the ground point
        that corresponds to the range, look direction, height above ellipsoid
        and state vector of the radar satellite. For more details, see
        `Geo Model PDF`_
        
        Parameters
        ----------
        X : `numpy.ndarray`, (3,)
            Three element array corresponding to the XYZ coordinates of the 
            ground point.
        range : `float`
            The range from the satellite to the point on the ground (m).
        u : `float`
            The look direction from the satellite to the point on the ground.
            This is the measured relative to the satellite velocity and is
            in the range [-1,1]. For braodside, or zero-Doppler, u=0.
        h : `float`
            The height of the ground point above the ellipsoid (m).
        sX : `numpy.ndarray`, (3,)
            Position component of satellite state vector.
        sV : `numpy.ndarray`, (3,)
            Velocity component of satellite state vector.
            
        Returns
        -------
        ndarray
            Output of the system of equations evaluated at the point X. This
            is a 3 element ndarray.
            
        """
        
        return np.array([((X/self.eS)**2).sum() 
                         - (X*X).sum()/(np.linalg.norm(X)-h)**2,
                         1.0-((X-sX)*(X-sX)).sum()/range**2,
                         np.dot(X - sX, sV) - range*np.linalg.norm(sV)*u])
                    

    def sysJac(self, X, range, h, sX, sV):
        """
        Jacobian of the System of equations to relate radar coordinates to 
        points on the ground
        
        Returns the Jacobian of the system of equations relating the ground 
        point that corresponds to the range, look direction, height above 
        ellipsoid and state vector of the radar satellite. For more details,
        see `Geo Model PDF`_
        
        Parameters
        ----------
        X : `numpy.ndarray`, (3,)
            Three element array corresponding to the XYZ coordinates of the 
            ground point.
        range : `float`
            The range from the satellite to the point on the ground (m).
        h : `float`
            The height of the ground point above the ellipsoid (m).
        sX : `numpy.ndarray`, (3,)
            Position component of satellite state vector.
        sV : `numpy.ndarray`, (3,)
            Velocity component of satellite state vector.
            
        Returns
        -------
        `numpy.ndarray`, (3,3)
            Jacobian of the system of equations evaluated at the point X. This
            is a 3x3 ndarray.
            
        """
        return np.array([2.0*(X/self.eS)/self.eS
                         + 2.0*X*h/(np.linalg.norm(X)-h)**3,
                         2.0*(sX-X)/range**2, sV])
        
        
    def computeECEF(self, sVec, u, range, h, X = None):
        """
        Computes the ECEF XYZ position from radar parameters
        
        Given the radar satellite state vector, the look direction of the
        radar beam, the range to the target point and the height of the
        target point above the ellipsoid, this function computes the ECEF
        Cartesian coordinates of the target point. The method used is 
        described `Geo Model PDF`_
        
        Parameters
        ----------
        sVec : `numpy.ndarray`, (6,)
            The radar satellite state vector.
        u : `float`
            Look direction in azimuth in range [-1,1]. For braodside, or 
            zero-Doppler, u=0.
        range : `float`
            Range to the target point (m).
        h : `float`
            Height of the target point above the ellipsoid (m).
        X : `numpy.ndarray`, (3,), optional
            Initial guess of the target point. The default is None.
            
        Returns
        -------
        X : `numpy.ndarray`, (3,)
            The estimated ground point in ECEF.
        error : `float`
            The error associated with the estimate(m).
            
        """
        sX = sVec[0:3]
        sV = sVec[3:6]
        
        if X is None:
            # Calculate the vector orthogonal to the position and velocity
            su = np.cross(sX, sV)
            su = su/np.linalg.norm(su)
            
            # Calculate a dummy variable
            srange = np.sqrt(range**2-(np.linalg.norm(sX) 
                                       - self.body.a/2.0 
                                       - self.body.b/2.0)**2 )
            
            # Generate an intial guess
            wP = sX - np.cos(u)*su*srange
            # wP = sX - su*srange
            X =  wP*(self.body.a/2.0 + self.body.b/2.0)/np.linalg.norm(wP)
        
        # Now apply the Newton Raphson method
        for j in np.arange(10):
            f = self.sysEq(X, range, u, h, sX, sV)
            J = self.sysJac(X, range, h, sX, sV)
            dX = np.dot(np.linalg.inv(J), f)
            X = X - dX
            error = np.linalg.norm(dX)
            if(error < 1.0e-6):
                break
        
        return (X, error)
       
    def computeLLHwithDEM(self, 
                          sVec, 
                          u, 
                          range, 
                          dem = None, 
                          geoidHeightFunction=None, 
                          X=None):
        """
        Computes the ECEF XYZ position from radar parameters
        
        Given the radar satellite state vector, the look direction of the
        radar beam, the range to the target point and a DEM, this function 
        computes the ECEF Cartesian coordinates of the target point. The 
        method used is described `Geo Model PDF`_
        
        Parameters
        ----------
        sVec : `numpy.ndarray`, (6,)
            The radar satellite state vector.
        u : `float`
            Look direction in azimuth in range [-1,1]. For braodside, or 
            zero-Doppler, u=0.
        range : `float`
            Range to the target point (m).
        DEM : :class:`~sip_tools.DEM.DEM.DEM`
            A DEM object to compute heights.
        geoidHeightFunction : function, optional
            A function to convert HAG to HAE. The default is None
        X : `numpy.ndarray`, (3,), optional
            Initial guess of the target point. The default is None.
            
        Returns
        -------
        X : `numpy.ndarray`, (3,)
            The estimated ground point in ECEF.
            
        """
        
        # Start with a height of zero
        demH = 0.0
        
        # Define a default dem
        # dem = dem or DEM()
        
        # Variable to hold potential geoid height compensation
        gHeight = 0.0
        
        # Define an array to convert from radians to degrees
        rad2deg = np.array([180.0/np.pi, 180.0/np.pi, 1.0])
        
        for k in np.arange(10):
            h = demH
            (X, error) = self.computeECEF(sVec, u, range, h, X)
            (crds, perror) = self.xyz2polar(X)
            crdsDeg = crds*rad2deg
            
            # Estimate the height at this point
            demH = dem.estimate(crdsDeg[0], crdsDeg[1])[0]
            if(np.isnan(demH)):
                demH = 0.0
                #return None,None,None
            
            # Convert HAG to HAE
            if(geoidHeightFunction and k==0):
                gHeight = geoidHeightFunction(crds[0], crds[1])
                
            demH += gHeight
            
            # Check for convergence
            if(np.abs(demH - h) < 1.0):
                break
            
        return (crdsDeg, X, np.abs(demH-h))
        
    def geoF(self, llh, X):
        """
        System of equations for calculating LLH from XYZ
        
        The system of equations for calculaging LLH from XYZ by using the
        Newton-Raphson method
        
        Parameters
        ----------
        llh : `numpy.ndarray`, (3,)
            Array of lat (dd), long (dd), HAE (m).
        X : `numpy.ndarray`, (3,)
            The current guess of the ECEF coordiantes.
            
        Returns
        -------
        `numpy.ndarray`, (3,)
            The output of the system of equations.
            
        """
        # Compute the flattening factor
        f = (self.body.a-self.body.b)/self.body.a
        lat = llh[0]
        lon = llh[1]
        h = llh[2]
        
        # Get the C constant
        C = 1.0/np.sqrt( np.cos(lat)**2 + (1.0-f)**2*np.sin(lat)**2 )
        S = (1.0-f)**2*C
        
        return np.array([(self.body.a*C+h)*np.cos(lat)*np.cos(lon) - X[0], 
                         (self.body.a*C+h)*np.cos(lat)*np.sin(lon) - X[1],
                         (self.body.a*S+h)*np.sin(lat) - X[2]])

    def geoJ(self, llh):
        """
        Jacobian of the system of equations for calculating LLH from XYZ
        
        The Jacobian of the system of equations for calculaging LLH from XYZ.
        This function is used by the Newton-Raphson method.
        
        Parameters
        ----------
        llh : `numpy.ndarray`, (3,)
            Array of lat (dd), long (dd), HAE (m).
            
        Returns
        -------
        `numpy.ndarray`, (3,3)
            The Jacobian 3x3.
            
        """
        # Compute the flattening factor
        f = (self.body.a-self.body.b)/self.body.a
        lat = llh[0]
        lon = llh[1]
        h = llh[2]
        
        # Get the C constant
        C = 1.0/np.sqrt( np.cos(lat)**2 + (1.0-f)**2*np.sin(lat)**2 )
        S = (1.0-f)**2*C
        dCdLat = (f-f**2/2.0)*np.sin(2.0*lat)*C**3
        dSdLat = (1.0-f)**2*dCdLat
        
        # Define the trig values
        clat = np.cos(lat)
        clon = np.cos(lon)
        slat = np.sin(lat)
        slon = np.sin(lon)
        
        # Define the Jacobian elements
        J11 = self.body.a*dCdLat*clat*clon - (self.body.a*C+h)*slat*clon
        J12 = -1.0*(self.body.a*C+h)*clat*slon
        J13 = clat*clon
        J21 = self.body.a*dCdLat*clat*slon - (self.body.a*C+h)*slat*slon
        J22 = (self.body.a*C+h)*clat*clon
        J23 = clat*slon
        J31 = self.body.a*dSdLat*slat + (self.body.a*S+h)*clat
        J32 = 0.0
        J33 = slat
        
        # Define the matrix
        return np.array([[J11, J12, J13], [J21, J22, J23], [J31, J32, J33]]) 
        
    def xyz2polar(self, X):
        """
        Convert ECEF XYZ coordinates to geographic coordinates EPSG:4326
        
        Use the Newton-Raphson method to compute the EPSG:4326 coordinates
        of an ECEF Cartesian XYZ point.
        
        Parameters
        ----------
        X : `numpy.ndarray`, (3,)
            Three component array of the ECEF Cartesian coordinates (all in m)
            to be converted into EPSG:4326.
            
        Returns
        -------
        llh : `numpy.ndarray`, (3,)
            The computed lat, long and height above ellipsoid (m).
        error : `float`
            Error in the calculation. The smaller the error, the better the
            convergence of the method. There are no units to this error as
            the coordinates in llh are of mixed type with the first two 
            coordinates in decimal degrees and the last in m. The error is
            hard coded to be less than 1.0e-9. That is, the L2 norm of the 
            difference between the esimated llh and the true llh is less than
            1e-9
            
        """
        # Convert to spherical polar
        llh = np.array([np.arctan(X[2]/np.sqrt(X[0]**2+X[1]**2)), 
                        np.arctan2(X[1],X[0]), 0.0])
        
            
        # Now apply the Newton Raphson method
        for j in np.arange(10):
            f = self.geoF(llh,X)
            J = self.geoJ(llh)
            dLLH = np.dot(np.linalg.inv(J), f)
            llh = llh - dLLH
            error = np.linalg.norm(dLLH)
            if(error < 1.0e-9):
                break
        
        return (llh, error)
    
    
    def slowTimefastTime2llh(self, 
                             slowTime, 
                             fastTime, 
                             sv=None, 
                             dem=None, 
                             u=0.0):
        # This function will try to compute the llh coordinate in WGS84 for
        # a given slowTime, fastTime, satellite and DEM. If the DEM is not provided
        # Then the default DTED2 DEM will be used
        
        # Get the satellite state vector at the slow time
        if not sv:
            print("Could not load the state vectors")
            return None, None, None
        
        # Set the default DEM as required
        # dem = dem or DEM()
        
        # Convert azimuth string to datetime
        UTC = datetime.datetime.strptime(slowTime, "%Y-%m-%dT%H:%M:%S.%fZ")
        
        # Compute the range
        rng = fastTime*self.c/2.0
        
        # Calculate the coordinates and return
        return (self.computeLLHwithDEM(sv.estimate(UTC), 
                                       u, 
                                       rng, 
                                       dem, 
                                       sv.geoidHeight), sv, dem)
    