import numpy as np

#from DEM.DEM import DEM
from measurement.measurement import WGS84 as eM
import datetime


class satGeometry:
    # Function to compute the ECEF coordinates
    c = 299792458.0
    orbitFile = u"r:/SARorbits.xml"
    
    def __init__(self):
        self.eS = np.array([eM.aE, eM.aE, eM.bE])

    # Compute the vector of functions
    def sysEq(self, X, range, u, h, sX, sV):
        # Return a vector of the value sof the system of 
        # euqations
        
        return np.array([((X/self.eS)**2).sum() - (X*X).sum()/(np.linalg.norm(X)-h)**2,
                    1.0-((X-sX)*(X-sX)).sum()/range**2,
                    np.dot(X - sX, sV) - range*np.linalg.norm(sV)*u])
                    

    def sysJac(self, X, range, h, sX, sV):
        # Define and return the Jacobian matrix
        return np.array([2.0*(X/self.eS)/self.eS+ 2.0*X*h/(np.linalg.norm(X)-h)**3,
        2.0*(sX-X)/range**2, sV])
        
        
    def computeECEF(self, sVec, u, range, h, X = None):
        sX = sVec[0:3]
        sV = sVec[3:6]
        
        if X is None:
            # Calculate the vector orthogonal to the position and velocity
            su = np.cross(sX, sV)
            su = su/np.linalg.norm(su)
            
            # Calculate a dummy variable
            srange = np.sqrt(range**2-(np.linalg.norm(sX) - eM.aE/2.0 - eM.bE/2.0)**2 )
            
            # Generate an intial guess
            wP = sX - np.cos(u)*su*srange
            # wP = sX - su*srange
            X =  wP*(eM.aE/2.0 + eM.bE/2.0)/np.linalg.norm(wP)
        
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
       
    def computeLLHwithDEM(self, sVec, u, range, dem = None, geoidHeightFunction=None, X=None):
        # Start with a height of zero
        demH = 0.0
        
        # Define a default dem
        dem = dem or DEM()
        
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
        # Compute the flattening factor
        f = (eM.aE-eM.bE)/eM.aE
        lat = llh[0]
        lon = llh[1]
        h = llh[2]
        
        # Get the C constant
        C = 1.0/np.sqrt( np.cos(lat)**2 + (1.0-f)**2*np.sin(lat)**2 )
        S = (1.0-f)**2*C
        
        return np.array([(eM.aE*C+h)*np.cos(lat)*np.cos(lon) - X[0], 
                                 (eM.aE*C+h)*np.cos(lat)*np.sin(lon) - X[1],
                                 (eM.aE*S+h)*np.sin(lat) - X[2]])

    def geoJ(self, llh):
        # Compute the flattening factor
        f = (eM.aE-eM.bE)/eM.aE
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
        J11 = eM.aE*dCdLat*clat*clon - (eM.aE*C+h)*slat*clon
        J12 = -1.0*(eM.aE*C+h)*clat*slon
        J13 = clat*clon
        J21 = eM.aE*dCdLat*clat*slon - (eM.aE*C+h)*slat*slon
        J22 = (eM.aE*C+h)*clat*clon
        J23 = clat*slon
        J31 = eM.aE*dSdLat*slat + (eM.aE*S+h)*clat
        J32 = 0.0
        J33 = slat
        
        # Define the matrix
        return np.array([[J11, J12, J13], [J21, J22, J23], [J31, J32, J33]]) 
        
    def xyz2polar(self, X):
        # Convert to spherical polar
        llh = np.array([np.arctan(X[2]/np.sqrt(X[0]**2+X[1]**2)), np.arctan2(X[1],X[0]), 0.0])
        
            
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
    
    
    def slowTimefastTime2llh(self, slowTime, fastTime, sv=None, dem=None, u=0.0):
        # This function will try to compute the llh coordinate in WGS84 for
        # a given slowTime, fastTime, satellite and DEM. If the DEM is not provided
        # Then the default DTED2 DEM will be used
        
        # Get the satellite state vector at the slow time
        if not sv:
            print("Could not load the state vectors")
            return None, None, None
        
        # Set the default DEM as required
        dem = dem or DEM()
        
        # Convert azimuth string to datetime
        UTC = datetime.datetime.strptime(slowTime, "%Y-%m-%dT%H:%M:%S.%fZ")
        
        # Compute the range
        rng = fastTime*self.c/2.0
        
        # Calculate the coordinates and return
        return self.computeLLHwithDEM(sv.estimate(UTC), u, rng, dem, sv.geoidHeight), sv, dem
    