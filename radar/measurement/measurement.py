import datetime
from math import asin, atan, atan2, pi, fabs
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.special import lpmn
from scipy import mat, multiply
import xml.etree.ElementTree as etree
import numpy as np
from numpy import meshgrid, arange, sqrt, zeros, cumprod, array, diag, matrix, cos, sin
import os
import math
import share.resource as res

egmFilePath = os.path.split(res.__file__)[0]

class measurement:
    def __init__(self, mTime=[], mData=[]):
        if(mTime):
            self.measurementTime.append(mTime)
            self.measurementData.append(mData)

    def add(self, mTime, mData):
        try:
            self.measurementTime.append(mTime)
            self.measurementData.append(np.array(mData))
        except AttributeError as aE:
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
#        minVal = 50.0
#        minK = 0
#        for k in range(0,len(self.measurementTime)):
#            deltaT = fabs((dtime-self.measurementTime[k]).total_seconds())
#            if(deltaT<minVal):
#                minVal = deltaT
#                minK = k
#        return minK

class attitude_vector(measurement):
    def estimate(self, dtime):
        minK = self.findNearest(dtime)
        y = self.measurementData[minK]
        return y

class GRS80():
    GM = 3.986005e14
    aE = 6378137.0
    bE = 6356752.3141
    siderealSeconds = 86164.099
    wE = 7.292115e-5
    J2 = 0.00108263
    J4 = -0.00000237091222
    J6 = 0.00000000608347
    J8 = -0.00000000001427

class WGS84():
    aE = 6378137.0
    bE = 6356752.3142
    
class EGM96():
    GM = 3.986004415e14
    aE = 6378136.3
    siderealSeconds = 86164.099
    wE = 7.292115e-5
    
class state_vector(measurement):
    constGRS80 = GRS80()
    constEGM96 = EGM96()
    constWGS84 = WGS84()
    Nharmonics = 20
    NharmonicsToLoad = 20
    egm96 = []
    
    
    def __init__(self, svFile=None, egmfile=None, egmCoeff=360):
        egmFile = egmfile or os.path.join(egmFilePath , "egm96.txt")
        if(os.path.exists(egmFile) and egmCoeff):
            try:
                self.loadSphericalHarmonics(egmFile, egmCoeff)
            except IOError as e:
                print(e)
                
        
        if(svFile):
            print("loading the state vector file:")
            print(svFile)
            self.readStateVectors(svFile)
            

    def xyz2polar(self, mData):
        a = self.constWGS84.aE
        b = self.constWGS84.bE
        f=(a-b)/a
        es=2.0*f-f*f;
        p=sqrt(mData[0]*mData[0]+mData[1]*mData[1])
        latitude=atan(mData[2]*a/p*b)
        ess=(a*a-b*b)/(b*b)
        for k in range(1,1000):
            la = atan2(mData[2]+ess*b*sin(latitude)**3,p-es*a*cos(latitude)**3)
            if(abs(latitude-la)<1.0e-9):
                break
            latitude = la
        longitude = atan2(mData[1],mData[0])
        Nphi=a/sqrt(1.0-es*sin(latitude)**2)
        hae=p/cos(latitude)-Nphi
        latitude = latitude/pi*180
        longitude = longitude/pi*180
        return latitude, longitude, hae

    def xyz2SphericalPolar(self, mData):
        r = sqrt(mData[0]*mData[0]+mData[1]*mData[1]+mData[2]*mData[2])
        latitude = asin(mData[2]/r)
        longitude = atan2(mData[1],mData[0])
        latitude = latitude/pi*180.0
        longitude = longitude/pi*180.0
        return latitude, longitude, r
    
    def llh2xyz(self, mData):
        # mData contains the lat, lon, height in degrees
        a = self.constWGS84.aE
        b = self.constWGS84.bE
        phi = mData[0]*pi/180.0
        lam = mData[1]*pi/180.0
        h = mData[2]
        
        N = a**2/sqrt(a**2*cos(phi)**2+b**2*sin(phi)**2)
        X = (N+h)*cos(phi)*cos(lam)
        Y = (N+h)*cos(phi)*sin(lam)
        Z = (b**2*N/a**2+h)*sin(phi)
        return np.array([X,Y,Z])
    
    def computeBroadsideToX(self, eta, X):
        for k in range(10):
            slaveX = self.estimate(eta)
            slaveV = self.satEQM(slaveX, 0.0)
            slX = slaveX[0:3]
            slV = slaveX[3:]
            slA = slaveV[3:]
            error = np.dot(slV,slX-X)/(np.dot(slA, slX-X) + np.dot(slV, slV))
            
            # Update the time variable
            eta = eta - datetime.timedelta(seconds=error)
            
            # Check to break from the loop. Recall that we only measure time
            # to the closest 1.0e-6 in the usec field
            if(abs(error) < 1.0e-6): 
                break
            
        return [eta, slaveX, error]
        
    def getDateTimeXML(self, XMLDateTimeElement):
        return datetime.datetime(int(XMLDateTimeElement.find('year').text),
                                 int(XMLDateTimeElement.find('month').text),
                                 int(XMLDateTimeElement.find('day').text),
                                 int(XMLDateTimeElement.find('hour').text),
                                 int(XMLDateTimeElement.find('minute').text),
                                 int(XMLDateTimeElement.find('sec').text),
                                 int(XMLDateTimeElement.find('usec').text))
    
    def loadSphericalHarmonics(self, filename, Nharmonics=None):
        self.egm96 = []

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
            self.egm96.append([int(fields[0]),int(fields[1]),float(fields[2]),float(fields[3])])
            row = f.readline()
            if(row):
                fields = row.split()
            else:
                break
        f.close()
        self.Nharmonics = self.NharmonicsToLoad
        return

    def selectEGMNew(self,idx):
        C = []
        S = []
        for x in self.egm96:
            if(int(x[0])==idx):
                C.append(float(x[2]))
                S.append(float(x[3]))
        return array([C,S]).T
        
    def selectEGM(self,idx):
        C = []
        S = []
        for x in self.egm96:
            if(int(x[0])==idx):
                C.append(float(x[2]))
                S.append(float(x[3]))
        return mat([C,S]).H
    
    def grs80radius(self, lat):
        ee = (self.constGRS80.aE/self.constGRS80.bE)**2 - 1.0
        return self.constGRS80.aE/sqrt(1.0 + ee*sin(lat)**2)
        
    def geoidHeight(self, lat, lon):
        # Compute the GRS80 equivalent radius
        r = self.grs80radius(lat)
        h = -(self.egm96Potential(r, lat, lon) - self.ellipsoidPotential(r, lat, lon))
        return h/self.egm96delR(r, lat, lon)
        
    def d2r(self, deg):
        return deg/180.0*pi
    
    def egm96PotentialOlder(self, r, lat, lon):
        GM = self.constEGM96.GM
        a = self.constEGM96.aE

        # Compute the coefficients for the gradient of the gravitational potential
        D = 1.0*GM/r
        normLegendreCoefficients = self.myLegendre(self.Nharmonics, sin(lat))
        cosines = [cos(l*lon) for l in range (0, self.Nharmonics+1)]
        sines = [sin(l*lon) for l in range (0, self.Nharmonics+1)]
        for k in range(2,self.Nharmonics+1):
            #mv=self.legendreNorm(k,sin(lat))
            mv=normLegendreCoefficients[k,0:(k+1)]
            egm = self.selectEGMNew(k)
            cs = sum([mv[l]*(cosines[l]*egm[l, 0] + sines[l]*egm[l,1]) for l in range(0,k+1)])
            D += 1.0*(GM/r)*cs*(a/r)**k

        return D 
        
    def egm96Potential(self, r, lat, lon):
        GM = self.constEGM96.GM
        a = self.constEGM96.aE

        # Compute the coefficients for the gradient of the gravitational potential
        D = 1.0*GM/r
        normLegendreCoefficients = self.myLegendre(self.Nharmonics, sin(lat))
        #rt = np.exp(1j*lon)
        #exprt = [rt**l for l in range (0, self.Nharmonics+1)]
        #cosines = np.real(exprt)
        #sines = np.imag(exprt)
        cosines = [cos(np.mod(l*lon, 2.0*np.pi)) for l in range (0, self.Nharmonics+1)]
        sines = [sin(np.mod(l*lon, 2.0*np.pi)) for l in range (0, self.Nharmonics+1)]
        for k in range(2,self.Nharmonics+1):
            egmIndex = int((k-2)*(k+3)/2)
            #mv=self.legendreNorm(k,sin(lat))
            mv=normLegendreCoefficients[k,0:(k+1)]
            #egm = self.selectEGMNew(k)
            cs = np.sum([mv[l]*(cosines[l]*self.egm96[egmIndex+l][2] + sines[l]*self.egm96[egmIndex+l][3]) for l in range(0,k+1)])
            D += 1.0*(GM/r)*cs*(a/r)**k

        return D

    def ellipsoidPotential(self, r, lat, lon):
        GM = self.constGRS80.GM
        a = self.constGRS80.aE
        
        J = [self.constGRS80.J2, self.constGRS80.J4, self.constGRS80.J6, self.constGRS80.J8]
        
        # Compute the coefficients of the harmonic expansion
        coeff = [-j/sqrt(2.0*l+1.0)*self.legendreNorm(l,sin(lat))[0,0]*(a/r)**l for j,l in zip(J, range(2,10,2))]
        
        # Add the very first term to the cumulative sum and multiple by physical constants
        D = GM/r*(1.0 + sum(coeff))

        return D 
    
    def egm96delR(self, r, lat, lon):
        GM = self.constEGM96.GM
        a = self.constEGM96.aE
        rSquare = r**2

        # Apply the formula
        D = -1.0*GM/rSquare
        
        # Compute the coefficients for the gradient of the gravitational potential
        normLegendreCoefficients = self.myLegendre(self.Nharmonics, sin(lat))
        cosines = [cos(l*lon) for l in range (0, self.Nharmonics+1)]
        sines = [sin(l*lon) for l in range (0, self.Nharmonics+1)]
        for k in range(2,self.Nharmonics+1):
            egmIndex = int((k-2)*(k+3)/2)
            mv=normLegendreCoefficients[k,0:(k+1)]
            cs = sum([mv[l]*(cosines[l]*self.egm96[egmIndex+l][2] + sines[l]*self.egm96[egmIndex+l][3]) for l in range(0,k+1)])
            D += -1.0*GM*cs*(k+1.0)*(a/r)**k/rSquare

        return D 
    
    def egm96delRdelR(self, r, lat, lon):
        GM = self.constEGM96.GM
        a = self.constEGM96.aE
        rCubed = r**3

        # Apply the formula
        D = 2.0*GM/rCubed
        
        # Compute the coefficients for the gradient of the gravitational potential
        normLegendreCoefficients = self.myLegendre(self.Nharmonics, sin(lat))
        cosines = [cos(l*lon) for l in range (0, self.Nharmonics+1)]
        sines = [sin(l*lon) for l in range (0, self.Nharmonics+1)]
        for k in range(2,self.Nharmonics+1):
            egmIndex = int((k-2)*(k+3)/2)
            mv=normLegendreCoefficients[k,0:(k+1)]
            cs = sum([mv[l]*(cosines[l]*self.egm96[egmIndex+l][2] + sines[l]*self.egm96[egmIndex+l][3]) for l in range(0,k+1)])
            D += GM*cs*(k+1.0)*(k+2.0)*(a/r)**k/rCubed

        return D 
    
    def egm96delROld(self, r, lat, lon):
        GM = self.constEGM96.GM
        a = self.constEGM96.aE
        rSquare = r**2

        # Compute the coefficients for the gradient of the gravitational potential
        sphCoeff = []
        normLegendreCoefficients = self.myLegendre(self.Nharmonics, sin(lat))
        for k in range(2,self.Nharmonics+1):
            #mv=self.legendreNorm(k,sin(lat))
            mv=matrix(normLegendreCoefficients[k,0:(k+1)])
            egm = self.selectEGM(k)
            cs = mat([[cos(l*lon), sin(l*lon)] for l in range(0,k+1)])
            innerSum = mv*multiply(cs,egm)*mat([1,1]).H
            sphCoeff.append(float(innerSum[0])/rSquare)

        # Apply the formula
        D = -1.0*GM/rSquare

        for C,k in zip(sphCoeff, range(2, self.Nharmonics+1)):
            D += -1.0*GM*C*(k+1.0)*(a/r)**k

        return D 
    
    # T corresponds to longitude    
    def egm96delT(self, r, lat, lon):
        GM = self.constEGM96.GM
        a = self.constEGM96.aE
        D = 0.0

        # Compute the coefficients for the gradient of the gravitational potential
        normLegendreCoefficients = self.myLegendre(self.Nharmonics, sin(lat))
        cosines = [cos(l*lon) for l in range (0, self.Nharmonics+1)]
        sines = [sin(l*lon) for l in range (0, self.Nharmonics+1)]
        for k in range(2,self.Nharmonics+1):
            egmIndex = int((k-2)*(k+3)/2)
            mv=normLegendreCoefficients[k,0:(k+1)]
            cs = sum([mv[l]*(-l*sines[l]*self.egm96[egmIndex+l][2] + l*cosines[l]*self.egm96[egmIndex+l][3]) for l in range(0,k+1)])
            D += GM*cs*(a/r)**k/r

        return D 
        
    def egm96delTdelT(self, r, lat, lon):
        GM = self.constEGM96.GM
        a = self.constEGM96.aE
        D = 0.0

        # Compute the coefficients for the gradient of the gravitational potential
        normLegendreCoefficients = self.myLegendre(self.Nharmonics, sin(lat))
        cosines = [cos(l*lon) for l in range (0, self.Nharmonics+1)]
        sines = [sin(l*lon) for l in range (0, self.Nharmonics+1)]
        for k in range(2,self.Nharmonics+1):
            egmIndex = int((k-2)*(k+3)/2)
            mv=normLegendreCoefficients[k,0:(k+1)]
            cs = sum([mv[l]*(-l**2*cosines[l]*self.egm96[egmIndex+l][2] - l**2*sines[l]*self.egm96[egmIndex+l][3]) for l in range(0,k+1)])
            D += GM*cs*(a/r)**k/r

        return D 
    
    def egm96delTdelR(self, r, lat, lon):
        GM = self.constEGM96.GM
        a = self.constEGM96.aE
        D = 0.0

        # Compute the coefficients for the gradient of the gravitational potential
        normLegendreCoefficients = self.myLegendre(self.Nharmonics, sin(lat))
        cosines = [cos(l*lon) for l in range (0, self.Nharmonics+1)]
        sines = [sin(l*lon) for l in range (0, self.Nharmonics+1)]
        for k in range(2,self.Nharmonics+1):
            egmIndex = int((k-2)*(k+3)/2)
            mv=normLegendreCoefficients[k,0:(k+1)]
            cs = sum([mv[l]*(-l*sines[l]*self.egm96[egmIndex+l][2] + l*cosines[l]*self.egm96[egmIndex+l][3]) for l in range(0,k+1)])
            D += -(k+1.0)*GM*cs*(a/r)**k/r**2

        return D 
    
    def egm96delTOld(self, r, lat, lon):
        GM = self.constEGM96.GM
        a = self.constEGM96.aE

        # Compute the coefficients for the gradient of the gravitational potential
        sphCoeff = []
        normLegendreCoefficients = self.myLegendre(self.Nharmonics, sin(lat))
        for k in range(2,self.Nharmonics+1):
            mv=matrix(normLegendreCoefficients[k,0:(k+1)])
            egm = self.selectEGM(k)
            cs = mat([[-l*sin(l*lon), l*cos(l*lon)] for l in range(0,k+1)])
            innerSum = mv*multiply(cs,egm)*mat([1,1]).H
            sphCoeff.append(float(innerSum[0])/r)

        # Apply the formula
        D = 0.0

        for C,k in zip(sphCoeff, range(2, self.Nharmonics+1)):
            D += 1.0*GM*C*(a/r)**k

        return D 

    # U corresponds to latitude
    def egm96delU(self, r, lat, lon):
        GM = self.constEGM96.GM
        a = self.constEGM96.aE
        u = sin(lat)
        v = cos(lat)
        D = 0.0

        # Compute the coefficients for the gradient of the gravitational potential
        normLegendreCoefficients = self.myLegendre(self.Nharmonics, sin(lat))
        cosines = [cos(l*lon) for l in range (0, self.Nharmonics+1)]
        sines = [sin(l*lon) for l in range (0, self.Nharmonics+1)]
        for k in range(2,self.Nharmonics+1):
            egmIndex = int((k-2)*(k+3)/2)
            mv1 = normLegendreCoefficients[k,0:(k+1)]
            mv2 = normLegendreCoefficients[(k-1),0:(k+1)]
            f1 = [mv1[l]*u*k for l in range(0, k+1)]
            f2 = [mv2[l]*sqrt((k**2-l**2)*(k+0.5)/(k-0.5)) for l in range(0, k+1)]
            
            cs = sum([(f1[l]-f2[l])*(cosines[l]*self.egm96[egmIndex+l][2] + sines[l]*self.egm96[egmIndex+l][3]) for l in range(0,k+1)])
            D+= v*GM*cs*(a/r)**k/(u**2-1.0)/r

        return D 

    def fkl(self, k, l):
        if k>l:
            return np.sqrt((k**2-l**2)*(k+0.5)/(k-0.5))
        else:
            return 0.0
    
    def egm96delUdelU(self, r, lat, lon):
        GM = self.constEGM96.GM
        a = self.constEGM96.aE
        u = sin(lat)
        v = cos(lat)
        D = 0.0

        # Compute the coefficients for the gradient of the gravitational potential
        normLegendreCoefficients = self.myLegendre(self.Nharmonics, sin(lat))
        cosines = [cos(l*lon) for l in range (0, self.Nharmonics+1)]
        sines = [sin(l*lon) for l in range (0, self.Nharmonics+1)]
            
        for k in range(2,self.Nharmonics+1):
            egmIndex = int((k-2)*(k+3)/2)
            mv1 = normLegendreCoefficients[k,0:(k+1)]
            mv2 = normLegendreCoefficients[(k-1),0:(k+1)]
            mv3 = normLegendreCoefficients[(k-2),0:(k+1)]
            f1 = [mv1[l]*(k**2*u**2-k)/v**2 for l in range(0, k+1)]
            f2 = [mv2[l]*2.0*(1.0-k)*u/v**2*self.fkl(k,l) for l in range(0, k+1)]
            f3 = [mv3[l]*self.fkl(k,l)*self.fkl(k-1,l)/v**2 for l in range(0, k+1)]
            
            cs = sum([(f1[l]+f2[l]+f3[l])*(cosines[l]*self.egm96[egmIndex+l][2] + sines[l]*self.egm96[egmIndex+l][3]) for l in range(0,k+1)])
            D += GM*cs*(a/r)**k/r

        return D
    
    def egm96delUdelT(self, r, lat, lon):
        GM = self.constEGM96.GM
        a = self.constEGM96.aE
        u = sin(lat)
        v = cos(lat)
        D = 0.0

        # Compute the coefficients for the gradient of the gravitational potential
        normLegendreCoefficients = self.myLegendre(self.Nharmonics, sin(lat))
        cosines = [cos(l*lon) for l in range (0, self.Nharmonics+1)]
        sines = [sin(l*lon) for l in range (0, self.Nharmonics+1)]
        for k in range(2,self.Nharmonics+1):
            egmIndex = int((k-2)*(k+3)/2)
            mv1=normLegendreCoefficients[k,0:(k+1)]
            mv2 = normLegendreCoefficients[(k-1),0:(k+1)]
            f1 = [mv1[l]*u*k for l in range(0, k+1)]
            f2 = [mv2[l]*sqrt((k**2-l**2)*(k+0.5)/(k-0.5)) for l in range(0, k+1)]
            
            cs = sum([(f1[l]-f2[l])*(-l*sines[l]*self.egm96[egmIndex+l][2] + l*cosines[l]*self.egm96[egmIndex+l][3]) for l in range(0,k+1)])
            D+= GM*v*cs*(a/r)**k/(u**2-1.0)/r

        return D 

    def egm96delUdelR(self, r, lat, lon):
        GM = self.constEGM96.GM
        a = self.constEGM96.aE
        u = sin(lat)
        v = cos(lat)
        D = 0.0

        # Compute the coefficients for the gradient of the gravitational potential
        normLegendreCoefficients = self.myLegendre(self.Nharmonics, sin(lat))
        cosines = [cos(l*lon) for l in range (0, self.Nharmonics+1)]
        sines = [sin(l*lon) for l in range (0, self.Nharmonics+1)]
        for k in range(2,self.Nharmonics+1):
            egmIndex = int((k-2)*(k+3)/2)
            mv1=normLegendreCoefficients[k,0:(k+1)]
            mv2 = normLegendreCoefficients[(k-1),0:(k+1)]
            f1 = [mv1[l]*u*k for l in range(0, k+1)]
            f2 = [mv2[l]*sqrt((k**2-l**2)*(k+0.5)/(k-0.5)) for l in range(0, k+1)]
            
            cs = sum([(f1[l]-f2[l])*(cosines[l]*self.egm96[egmIndex+l][2] + sines[l]*self.egm96[egmIndex+l][3]) for l in range(0,k+1)])
            D+= -(k+1.0)*v*GM*cs*(a/r)**k/(u**2-1.0)/r**2

        return D 

    def egm96delUOld(self, r, lat, lon):
        GM = self.constEGM96.GM
        a = self.constEGM96.aE
        u = sin(lat)

        # Compute the coefficients for the gradient of the gravitational potential
        sphCoeff = []
        normLegendreCoefficients = self.myLegendre(self.Nharmonics, sin(lat))
        for k in range(2,self.Nharmonics+1):
            mv1=matrix(normLegendreCoefficients[k,0:(k+1)])
            mv2 = mat(matrix(normLegendreCoefficients[(k-1),0:k]).tolist()[0] + [0.0])
            f1 = mat([u*k for l in range(0, k+1)])
            f2 = mat([sqrt((k**2-l**2)*(k+0.5)/(k-0.5)) for l in range(0, k+1)])
            egm = self.selectEGM(k)
            cs = mat([[cos(l*lon), sin(l*lon)] for l in range(0,k+1)])
            innerSum = (multiply(mv1, f1)-multiply(mv2, f2))*multiply(cs,egm)*mat([1,1]).H
            sphCoeff.append(float(innerSum[0])/r/(u**2-1.0))

        # Apply the formula
        D = 0.0

        for C,k in zip(sphCoeff, range(2, self.Nharmonics+1)):
            D += 1.0*GM*C*(a/r)**k

        return D 

    def Jsx(self, X):
        r = np.linalg.norm(X[0:3])
        p = np.sqrt(X[0]**2 + X[1]**2)
        
        return np.array([X/r, 
                         [-X[0]*X[2]/r**2/p, -X[1]*X[2]/r**2/p, p/r**2], 
                         [-X[1]/p**2, X[0]/p**2, 0.0]
                        ])
    
    def secondDtve(self, X):
        x = X[0]
        y = X[1]
        z = X[2]
        
        r = np.linalg.norm(X[0:3])
        p = np.sqrt(x**2 + y**2)
        
        Jtr = 1.0/r*np.eye(3) - 1.0/r**3*np.outer(X[0:3],X[0:3])
#        Jtu = np.array([[-z*((r*y)**2-2.0*(x*p)**2), x*y*z*(2.0*p**2+r**2), -x*(p**4-(p*z)**2)],
#                        [x*y*z*(2.0*p**2+r**2), -z*((r*x)**2-2.0*(y*p)**2), -y*(p**4-(z*p)**2)],
#                        [-x*(p**4-(p*z)**2), -y*(p**4-(z*p)**2), -2.0*z*p**4]])/p**3/r**4
    
        Jtu = np.array([[-z*((r*y)**2-2.0*(x*p)**2), x*y*z*(2.0*p**2+r**2), -x*(p**4-(p*z)**2)],
                        [x*y*z*(2.0*p**2+r**2), -z*((r*x)**2-2.0*(y*p)**2), -y*(p**4-(z*p)**2)],
                        [-x*(p**4-(p*z)**2), -y*(p**4-(z*p)**2), -2.0*z*p**4]])/p**3/r**4
    
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
        return reshuffle.dot(np.array(list(Jtr)+list(Jtu)+list(Jtt)))
    
        
    def satEQM(self,X,t):
        # Define some constants
        wE = self.constGRS80.wE
        
        # Transform to lat/long/r spherical polar
        llh = self.xyz2SphericalPolar(X)
        lat = llh[0]/180.0*pi
        lon = llh[1]/180.0*pi

        # Compute the norm
        r = llh[2]
        p = np.sqrt(X[0]**2 + X[1]**2)
        Xp = mat(X[0:3])
        Xv = mat(X[3:])
        
        # Compute the component of the gradient due to r, theta, phi
        delVdelR = self.egm96delR(r, lat, lon)
        delVdelT = self.egm96delT(r, lat, lon)
        delVdelU = self.egm96delU(r, lat, lon)


        # Calculate factors to transform sperical polar to Cartesian derivatives
        delVdelRX = delVdelR/r*mat(X[0:3])
        delVdelUX = delVdelU*mat([-X[0]*X[2]/r**2/p, -X[1]*X[2]/r**2/p, p/r**2])
        delVdelTX = delVdelT*mat([-X[1]/p**2, X[0]/p**2, 0.0])


        # Compute some shifting matrices
        I2 = mat([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        Q2 = mat([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        
        XM = mat([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        VM = mat([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        
        
        # Do the calculation
        return np.array(X[3:]*XM + (wE**2*Xp*I2 + 2.0*wE*Xv*Q2 + delVdelRX + delVdelTX + delVdelUX)*VM).flatten()
    
    def isatEQM(self, t, X):
        return self.satEQM(X,t)
    
    def oldExpandedState(self,X,t):
        # Define some constants
        wE = self.constGRS80.wE
        
        # Transform to lat/long/r spherical polar
        llh = self.xyz2SphericalPolar(X)
        lat = llh[0]/180.0*pi
        lon = llh[1]/180.0*pi

        # Compute the norm
        r = llh[2]
        p = X[0]**2 + X[1]**2
        Xp = mat(X[0:3])
        Xv = mat(X[3:])
        
        # Compute the component of the gradient due to r, theta, phi
        delVdelR = self.egm96delR(r, lat, lon)
        delVdelT = self.egm96delT(r, lat, lon)
        delVdelU = self.egm96delU(r, lat, lon)


        # Calculate factors to transform sperical polar to Cartesian derivatives
        delVdelRX = delVdelR/r*mat(X[0:3])
        delVdelUX = delVdelU*mat([-X[0]*X[2]/r**3, -X[1]*X[2]/r**3, p/r**3])
        delVdelTX = delVdelT*mat([-X[1]/p, X[0]/p, 0.0])
        inertial_ddX = delVdelRX + delVdelTX + delVdelUX
        print(inertial_ddX)


        # Compute some shifting matrices
        I2 = mat([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        Q2 = mat([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])        
        
        # Do the calculation
        ecef_X = X[:3]
        ecef_dX = X[3:]
        ecef_ddX = wE**2*Xp*I2 + 2.0*wE*Xv*Q2 + inertial_ddX
        #ecef_dddX = -wE**3*Xp*Q2 + 3.0*wE**2*Xv*I2 + 3.0*wE*ecef_ddX*Q2
        #print("DEBUG")
        #print(ecef_dddX)
        inertial_dX = Xv-wE*Xp*Q2
        wO = np.linalg.norm(inertial_dX)/np.linalg.norm(Xp)
        ecef_dddX = 2.0*wE**3*Xp*Q2 - 3.0*wE**2*Xv*I2 + 3.0*wE*inertial_ddX*Q2 - wO**2*(Xv-wE*Xp*Q2)
        
        return np.array([ecef_X,
                         ecef_dX,
                         np.asarray(ecef_ddX).squeeze(),
                         np.asarray(ecef_dddX).squeeze()])
    
    def secondSphericalDerivative(self, r, lat, lon):
        # Calculate the U Hessian matrix
        h11 = self.egm96delRdelR(r, lat, lon)
        h12 = self.egm96delUdelR(r, lat, lon)
        h13 = self.egm96delTdelR(r, lat, lon)
        h22 = self.egm96delUdelU(r, lat, lon)
        h23 = self.egm96delUdelT(r, lat, lon)
        h33 = self.egm96delTdelT(r, lat, lon)
        return np.array([[h11, h12, h13],
                         [h12, h22, h23],
                         [h13, h23, h33]])
        
    def expandedState(self,X,t):
        # Define some constants
        wE = -self.constGRS80.wE
        
        # Transform to lat/long/r spherical polar
        llh = self.xyz2SphericalPolar(X)
        lat = llh[0]/180.0*pi
        lon = llh[1]/180.0*pi

        # Compute the norm
        r = llh[2]
        ecef_X = X[:3]
        ecef_dX = X[3:]
        
        # Compute the component of the gradient due to r, theta, phi
        d0 = self.egm96delR(r, lat, lon)
        d1 = self.egm96delU(r, lat, lon)
        d2 = self.egm96delT(r, lat, lon)
        
        #print("Inertial spherical acceleration")
        inertial_ddS = np.array([d0, d1, d2])
        #print(inertial_ddS)
        
        # Compute an expanded ddS
        #print("First derivative w.r.t. spherical polar:")
        exp_ddS = np.array([[d0, d1, d2, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, d0, d1, d2, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, d0, d1, d2]])
        #print(exp_ddS)
        
        #print("Second derivative w.r.t. rxixj")
        dtwosdx = self.secondDtve(X)
        #print(dtwosdx)
    
        # Calculate the U Hessian matrix
#        h11 = self.egm96delRdelR(r, lat, lon)
#        h12 = self.egm96delUdelR(r, lat, lon)
#        h13 = self.egm96delTdelR(r, lat, lon)
#        h22 = self.egm96delUdelU(r, lat, lon)
#        h23 = self.egm96delUdelT(r, lat, lon)
#        h33 = self.egm96delTdelT(r, lat, lon)
        Hu = self.secondSphericalDerivative(r, lat, lon)
        #print(Hu)



        # Compute some shifting matrices
        I2 = np.array([[1.0, 0.0, 0.0], 
                       [0.0, 1.0, 0.0], 
                       [0.0, 0.0, 0.0]])
        Q2 = np.array([[0.0, -1.0, 0.0], 
                       [1.0, 0.0, 0.0], 
                       [0.0, 0.0, 0.0]])        
        
        # Calculate factors to transform sperical polar to Cartesian derivatives
        J = self.Jsx(ecef_X)
        JT = J.T
        
        # Print the second derivatives
        #print("Jacobian")
        #print(J)
        F = exp_ddS.dot(dtwosdx) + JT.dot(Hu.dot(J))
        #print("F")
        #print(exp_ddS.dot(dtwosdx))
        #print(JT.dot(Hu.dot(J)))
        #print(F)
        #print("Inertial velocity:")
        inertial_dX = ecef_dX - wE*np.dot(Q2, ecef_X)
        #print(inertial_dX)
        #print("Inertial acceleration:")
        inertial_ddX = np.dot(JT, inertial_ddS)
        #print(inertial_ddX)
        #print("New inertial dddX:")
        inertial_dddX = F.dot(inertial_dX)
        #print(inertial_dddX)

        # Do the calculation
        ecef_ddX = wE**2*np.dot(I2,ecef_X) + 2.0*wE*np.dot(Q2, ecef_dX) + inertial_ddX
        
        
        #print("DEBUG")
        #print(ecef_dddX)
        wO = np.linalg.norm(inertial_dX)/np.linalg.norm(ecef_X)
        #print("Inertial Jerk approximation")
        #print(-wO**2*inertial_dX)
        #ecef_dddX = (2.0*wE**3*np.dot(Q2, ecef_X) 
        #             - 3.0*wE**2*np.dot(I2, ecef_dX) 
        #             + 3.0*wE*np.dot(Q2, inertial_ddX) 
        #             - wO**2*inertial_dX)
        
        ecef_dddX = (2.0*wE**3*np.dot(Q2, ecef_X) 
             - 3.0*wE**2*np.dot(I2, ecef_dX) 
             + 3.0*wE*np.dot(Q2, inertial_ddX) 
             + inertial_dddX)
        
        
        return np.array([ecef_X,
                         ecef_dX,
                         ecef_ddX,
                         ecef_dddX])
    
    def ratioOfFactorial(self, n, m):
        if m==0:
            return 1.0
        return reduce(lambda x, y: float(x)/float(y), [1] + range(n+1-m,n+1+m))
    
    def myLegendre(self, n0, t):
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
        B[idx] = sqrt((2*N[idx]+1)*(N[idx]+M[idx]-1)*(N[idx]-M[idx]-1)/((N[idx]-M[idx])*(N[idx]+M[idx])*(2*N[idx]-3)))

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
        
        
    
    def legendreNorm(self,n,x):
        yval = lpmn(n,n,x)[0]
        y = []
        for m in range(0,n+1):
            y.append((-1)**m*sqrt((2.0*n+1.0)*self.ratioOfFactorial(n,m))*yval[m,n])

        return mat(y)

    def toInertial(self,mData, t):
        w = self.constWGS84.wE;
        X = mat([[cos(w*t),-sin(w*t),0.0],[sin(w*t),cos(w*t),0.0],[0.0,0.0,1.0]])
        V = w*mat([[-sin(w*t),-cos(w*t),0.0],[cos(w*t),-sin(w*t),0.0],[0.0,0.0,0.0]])
        ix = X*mat(mData[0:3]).H
        iv = X*mat(mData[3:6]).H + V*mat(mData[0:3]).H
        ivect = []
        for pos in ix:
            ivect.append(float(pos))
        for vel in iv:
            ivect.append(float(vel))
        return ivect

    def toECEF(self,mData, t):
        w = self.constWGS84.wE;
        X = mat([[cos(w*t),sin(w*t),0.0],[-sin(w*t),cos(w*t),0.0],[0.0,0.0,1.0]])
        V = w*mat([[-sin(w*t),cos(w*t),0.0],[-cos(w*t),-sin(w*t),0.0],[0.0,0.0,0.0]])
        ix = X*mat(mData[0:3]).H
        iv = X*mat(mData[3:6]).H + V*mat(mData[0:3]).H
        ivect = []
        for k in range(0,3):
            ivect.append(float(ix[k]))
        for k in range(0,3):
            ivect.append(float(iv[k]))
        return ivect

    def integrate(self, t, k, t_eval=None):
        if t_eval is None:
            y = solve_ivp(self.isatEQM, t, self.measurementData[k])
        else:
            y = solve_ivp(self.isatEQM, t, self.measurementData[k], t_eval=t_eval)
        return y
#        istate = []
#        for state, it in zip(y,t):
#            istate.append(state)
#        return istate
    
#    def integrate(self, t, k):
#        y = odeint(self.satEQM, self.measurementData[k], t)
#        istate = []
#        for state, it in zip(y,t):
#            istate.append(state)
#        return istate
        
#    def estimateTimeRange(self, dtime):
#        mnTime = sum([(tm - dtime[0]).total_seconds() for tm in dtime])/len(dtime)
#        minK = self.findNearest(dtime[0] + datetime.timedelta(seconds=mnTime))
#        integrationTimes = [(dt - self.measurementTime[minK]).total_seconds() for dt in dtime]
#        
#        # getintegration times
#        pTimes = [0.0] + [tm for tm in integrationTimes if tm >= 0]
#        nTimes = [tm for tm in integrationTimes if tm < 0] + [0.0]
#        nTimes.reverse()
#        
#        pState = []
#        nState = []
#        
#        # Integrate
#        if(len(pTimes) > 1):
#            pState = self.integrate(pTimes, minK)[1:]
#            
#        if(len(nTimes) > 1):
#            nState = self.integrate(nTimes, minK)[1:]
#            nState.reverse()
#            
#        # Create and return the output
#        return nState + pState
        
    def estimateTimeRange(self, dtime, integrationTimes = None):
        if type(dtime[0]) == datetime.datetime:
            mnTime = sum([(tm - dtime[0]).total_seconds() for tm in dtime])/len(dtime)
            minK = self.findNearest(dtime[0] + datetime.timedelta(seconds=mnTime))
        else:
            mnTime = sum([(tm - dtime[0])/np.timedelta64(1,'s') for tm in dtime])/len(dtime)
            minK = self.findNearest(dtime[0] + np.timedelta64(int(1e9*mnTime), 'ns'))
            
        #print(self.measurementTime[minK])
        if integrationTimes is None:
            if type(self.measurementTime[0]) == datetime.datetime:
                integrationTimes = [(dt - self.measurementTime[minK]).total_seconds() for dt in dtime]
            else:
                integrationTimes = [(dt - self.measurementTime[minK])/np.timedelta64(1,'s') for dt in dtime]
                
        
        # Perform the integration
        #t_min = np.min(integrationTimes)
        #t_max = np.max(integrationTimes)
        #sol = self.integrate([t_min, t_max], minK, t_eval = integrationTimes)
        #return sol.y, sol.t
        
        # getintegration times
        pTimes = [0.0] + [tm for tm in integrationTimes if tm > 0]
        nTimes = [tm for tm in integrationTimes if tm < 0] + [0.0]
        nTimes.reverse()
        
        #pState = []
        #nState = []
           
        if(len(nTimes) > 1):
            #print(nTimes)
            nState = self.integrate([np.max(nTimes), np.min(nTimes)], minK, t_eval = nTimes)
            #print(nState.t[0:5])
            #print(nState.y[0:3,0:3].T)
            #nState.reverse()
            
        # Integrate
        if(len(pTimes) > 1):
            #print(pTimes)
            pState = self.integrate([np.min(pTimes), np.max(pTimes)], minK, t_eval = pTimes)
            #print(pState.t[0:5])
         
            
        # Create and return the output
        if 0.0 in integrationTimes:
            t_array = np.concatenate((np.flipud(nState.t[1:]), pState.t))
            y_array = np.concatenate((np.fliplr(nState.y[:,1:]), pState.y), axis=1).T
        else:
            t_array = np.concatenate((np.flipud(nState.t[1:]), pState.t[1:]))
            y_array = np.concatenate((np.fliplr(nState.y[:,1:]), pState.y[:,1:]), axis=1).T
            
        #print("Time array length avant integration: %d" % len(dtime))
        #print("Time array length apres integration: %d" % len(t_array))
        return y_array
    
    def estimate(self, dtime):
        minK = self.findNearest(dtime)
        if type(self.measurementTime[0]) == datetime.datetime:
            dT = [0.0, (dtime - self.measurementTime[minK]).total_seconds()]
        else:
            dT = [0.0, (dtime - self.measurementTime[minK])/np.datetime64(1,'s')]
        sol = self.integrate(dT,minK)
        if not dtime in self.measurementTime:
            self.add(dtime, sol.y[:,-1])
        return sol.y[:,-1]

#    def estimate(self, dtime):
#        minK = self.findNearest(dtime)
#        dT = [0.0, (dtime - self.measurementTime[minK]).total_seconds()]
#        y = self.integrate(dT,minK)
#        if not dtime in self.measurementTime:
#            self.add(dtime, y[-1])
#        return y[-1]
        
class state_vector_TSX(state_vector):
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
            self.add(datetime.datetime.strptime(svTime.text, '%Y-%m-%dT%H:%M:%S.%f'), 
              [float(x.text),float(y.text),float(z.text),float(vx.text),float(vy.text),float(vz.text)])
        return

class state_vector_Radarsat(state_vector):
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
        
        for sv,x,y,z,vx,vy,vz in zip(stateVectorTimeElements,svX,svY,svZ,svVX,svVY,svVZ):
            self.add(self.getDateTimeXML(sv), [float(x.text),float(y.text),float(z.text),float(vx.text),float(vy.text),float(vz.text)])
        return
        
class state_vector_RSO(state_vector):
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
            self.add(refDate + datetime.timedelta(days=math.ceil(day), seconds = secsSinceMidnight),
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
    def readStateVectors(self, EODFile, desiredStartTime=None, desiredStopTime=None):
        xmlroot = etree.parse(EODFile).getroot()
        osvlist = xmlroot.findall(".//OSV")

        for osv in osvlist:
            utcElement = osv.find(".//UTC")
            datetime.datetime
            utc = datetime.datetime.strptime(utcElement.text.split('=')[1],'%Y-%m-%dT%H:%M:%S.%f')
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

        sTime = datetime.datetime.strptime(startTime.text[0:27], '%d-%b-%Y %H:%M:%S.%f')
        tD = float(timeDelta.text)
        sTimes = [sTime+datetime.timedelta(seconds=tD*r) for r in range(len(svX))]

        # Populate the local data
        for sv,x,y,z,vx,vy,vz in zip(sTimes,svX,svY,svZ,svVX,svVY,svVZ):
            self.add(sv, [float(x.text),
                          float(y.text),
                          float(z.text),
                          float(vx.text),
                          float(vy.text),
                          float(vz.text)])
        return
