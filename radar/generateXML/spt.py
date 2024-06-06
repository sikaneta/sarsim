#%%
""" This python program should generate the config files for burst mode multi 
    or single channel data.
    The program generates an xml file to be processed in MATLAB """

import numpy as np
import json
from scipy.constants import c
import os
import matplotlib.pyplot as plt

from utils.cjson import json2dict
from measurement.measurement import state_vector
from scipy.optimize import minimize

#%% Coefficients file
# roseLxsl = r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\RoseL\ROSE-UM-ADSF-SAR-1001590382_Iss.01_Antenna_Model_Installation\ROSEL_AntennaModel\Inputs\ROSEL_DPNear_mid.xlsx"
# sentinelNG = r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\Sentinel-NG\ISRR\SW01_aresys_Instrument Performance Mathematical Models including software\S1NG_pfsoftware\Inputs\S1NG_CP IW Ish.xlsx"
#%%
class sptRadar:
    
    def __init__(self, pool):
        self.pool = pool
        for k, sD in enumerate(pool["signalData"]):
            sD["baseline"] = self.baseline(0,k)
            
    def elementFactor(self, u, v, pol="H"):
        pool = self.pool
        epattern = pool["instrument"]["elementPattern"]
        return np.outer(np.interp(v, epattern["v"], epattern[pol]["elevation"]), 
                     np.interp(u, epattern["u"], epattern[pol]["azimuth"]))
    
    def patternFromConfiguration(self, MU, MV, component):
        if component["weights"]["type"] != "svd":
            return None
        fn = lambda s_elem: s_elem["s"]*np.outer(MV.dot(s_elem["u"]), 
                                                 s_elem["vh"].dot(MU))
        
        normalization = 1.0
        return sum(map(fn, component["weights"]["components"]))/normalization
    
    def computeStateVectors(self, channel = 0):
        orbit = self.pool["platform"]["orbit"]
        sv = state_vector()
        if orbit["type"] == "StateVector":
            if orbit["frame"] == "ECEF":
                sv.add(np.datetime64(orbit["time"]), 
                       np.array(orbit["xyzVxVyVz"]))
        rd = pool["signalData"][channel]
        nSamples = rd['acquisition']['numAzimuthSamples']
        deltaT = 1.0/rd['acquisition']['prf']
        sampleTimes = [sv.measurementTime[0] + 
                       np.timedelta64(int(k*deltaT*1e9), 'ns') 
                       for k in np.arange(nSamples)]
        svc = sv.estimateTimeRange(sampleTimes)
        for tm,posvel in zip(sampleTimes, svc):
            sv.add(tm, posvel)
        self.pool["platform"]["stateVectors"] = sv
            
    def arrayFactor(self, u, v, channel = 0):
        pool = self.pool
        antenna = pool["instrument"]["array"]
        wavenumber = 2*np.pi/c*pool["instrument"]["carrierFrequency"]
        UPos = antenna["azimuthPositions"]
        VPos = antenna["elevationPositions"]
        
        MU = np.exp(-1j*wavenumber*np.einsum("i,j -> ij", UPos, u))
        MV = np.exp(-1j*wavenumber*np.einsum("i,j -> ij", v, VPos))
        
        configs = pool["signalData"][channel]["radarConfiguration"]
        tPatt = self.patternFromConfiguration(MU, MV, configs["tx"])
        rPatt = self.patternFromConfiguration(MU, MV, configs["rx"])
        
        return tPatt, rPatt
    
    def baseline(self, m, n, v0 = [0.0]):
        u = self.pool["instrument"]["elementPattern"]["u"]
        p1 = self.twoWayAzimuthPatterns(u, v0 = v0, aziChannel = m)
        p2 = self.twoWayAzimuthPatterns(u, v0 = v0, aziChannel = n)
        intf = p1*p2.conj()
        wv = c/self.pool["instrument"]["carrierFrequency"]
        N = max(len(intf), 8192)
        ft = np.fft.fft(intf, N)
        idx = np.argmax(np.abs(ft))
        d0 = wv*idx/2/N/(u[1]-u[0])
        def J(d):
            return np.sum(intf*np.exp(-1j*4*np.pi*d/wv*np.array(u))).imag**2
        Jmin = minimize(J, d0, method='Nelder-Mead')
        if Jmin.success:
            return Jmin.x[0]
        else:
            return None
        
    def twoWayAzimuthPatterns(self, u, v0 = [0.0], aziChannel=0):
        u = self.pool["instrument"]["elementPattern"]["u"]
        pol = {"H": self.elementFactor(u, v0, pol="H"),
               "V": self.elementFactor(u, v0, pol="V")}
        sD = self.pool["signalData"][aziChannel]
        aPtx, aPrx = self.arrayFactor(u, v0, channel=aziChannel)
        txPol = sD["radarConfiguration"]["tx"]["polarization"]
        rxPol = sD["radarConfiguration"]["rx"]["polarization"]
        
        return aPtx*aPrx*pol[txPol]*pol[rxPol]

    def plotSpaceTime(self, N=20):
        va = np.linalg.norm(self.pool["platform"]["orbit"]["xyzVxVyVz"][3:])
        radar = self.pool["signalData"]
        number_CHANNELS = len(radar)
        pos = np.zeros((number_CHANNELS, N))
        tme = np.zeros((number_CHANNELS, N))
        ref_time = np.datetime64(radar[0]["acquisition"]["startTime"])
        
        leg_string = []
        for k, rd in enumerate(radar):
            t = np.arange(N)/rd['acquisition']['prf']
            sdelay = (np.datetime64(rd["acquisition"]["startTime"]) - ref_time)/np.timedelta64(1,'s')
            baseline = self.baseline(0,k)
            tme[k,:] = t+sdelay
            pos[k,:] = baseline + (t+sdelay)*va
            leg_string.append('channel %d: PC = %0.3f (m)' % (k,baseline))
            leg_string.append('channel %d: x-position' % k)
            
        symbol_ARRAY = ['x','o','+','d','^','v','.','p','h','>','<','s','d','*']
        colors_ARRAY = [(206.0/256, 113.0/256, 200.0/256),
                        (222.0/256, 128.0/256, 91.0/256),
                        (184.0/256, 54.0/256, 50.0/256),
                        (183.0/256, 207.0/256, 105.0/256),
                        (216.0/256, 237.0/256, 217.0/256),
                        (222.0/256, 128.0/256, 91.0/256),
                        (184.0/256, 54.0/256, 50.0/256),
                        (183.0/256, 247.0/256, 105.0/256),
                        (216.0/256, 37.0/256, 217.0/256),
                        (222.0/256, 18.0/256, 91.0/256),
                        (184.0/256, 54.0/256, 50.0/256),
                        (183.0/256, 207.0/256, 105.0/256),
                        (16.0/256, 237.0/256, 217.0/256),
                        (206.0/256, 213.0/256, 59.0/256)]
        
        plt.figure()
        sN = len(symbol_ARRAY)
        cN = len(colors_ARRAY)
        for k in range(number_CHANNELS):
            plt.plot(pos[k,:], tme[k,:], symbol_ARRAY[k%sN], color=colors_ARRAY[k%cN])
            plt.plot(pos[k,:], np.zeros(tme[k].shape), symbol_ARRAY[k%sN], color=colors_ARRAY[k%cN])
        plt.grid(True)
        plt.xlabel('Sample position in space (m)')
        plt.ylabel('Sample time (s)')
        plt.show()    

#%%
folder = os.path.join(r"C:\Users",
                      r"ishuwa.sikaneta",
                      r"OneDrive - ESA",
                      r"Documents",
                      r"ESTEC",
                      r"RoseL")
filename = "RoseL_Swath1_HH_IW.json"
with open(os.path.join(folder, filename), "r") as f:
    pool = json2dict(json.loads(f.read()))
    
#%%
sptN = sptRadar(pool)

#%%
sptN.computeStateVectors(channel=0)

#%% Plot the antenna pattern
u = pool["instrument"]["elementPattern"]["u"]
v = [0]

#%% Define a function to compute the azimuth two-way antenna pattern
def twoWayAzimuthPattern(u,
                         v0,
                         spt,
                         aziChannel = 0,
                         polarization = "HH"):
    aPtx, aPrx = spt.arrayFactor(u, [v0], channel=aziChannel)
    pol = {"H": spt.elementFactor(u, [v0], pol="H"),
           "V": spt.elementFactor(u, [v0], pol="V")}
    
    return aPtx*aPrx*pol[polarization[0]]*pol[polarization[1]]
    
#%%
twPtH1 = twoWayAzimuthPattern(u,0,sptN,aziChannel=0,polarization="HH")
twPtH2 = twoWayAzimuthPattern(u,0,sptN,aziChannel=2,polarization="HH")
#twPtH = aPtx*aPrx*ePtH*ePtH
#twPtV = aPtx*aPrx*ePtV*ePtV
#twPt = ePtH#*ePtH

#%%
plt.figure()
plt.plot(u, 20*np.log10(np.abs(twPtH1[0,:])))
plt.plot(u, 20*np.log10(np.abs(twPtH2[0,:])))
plt.grid()
plt.show()

#%%
plt.figure()
plt.plot(u, np.unwrap(np.angle(twPtH1[0,:]*twPtH2[0,:].conj())))
plt.grid()
plt.show()



#%%
intf = twPtH1[0,:]*twPtH2[0,:].conj()
N = max(len(intf), 8192)
ft = np.fft.fft(twPtH1[0,:]*twPtH2[0,:].conj(), N)
idx = np.argmax(np.abs(ft))
wavelength = c/pool["instrument"]["carrierFrequency"]
d = wavelength*idx/N/(u[1]-u[0])
print(d)

#%%
plt.figure()
plt.plot(np.abs(ft))
plt.show()