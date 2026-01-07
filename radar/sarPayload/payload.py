# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 15:14:37 2025

@author: Ishuwa.Sikaneta
"""
import json
from orbit.geometry import perspective
from space.planets import venus, earth
import numpy as np
from scipy.constants import c, Boltzmann
from sarPayload.pulsedGeometry import diamondFigure as dF
from orbit.kepler import svFromJsonArgs
import plotly.graph_objects as go
import scipy.integrate as integrate
import pandas as pd
from plot.diamond import antenna
import portion as P

#%% Define a class for computing various things
class sar:

    perf_colors = {
        "NESZ": "#00FFFF", 
        "range ambiguity": "#FF1493", 
        "azimuth ambiguity": "#FF00FF", 
        "MNR": "#FFA500"
    }
    
    def __init__(self, 
                 source = "generic.json",
                 sys_tag= "EuroSAR3_2", 
                 state_tag = "low",
                 off_nadir = np.radians(np.arange(5,55,0.01))):
        
        self.off_nadir = off_nadir
        radar = {"source": source,
                 "svtag": state_tag,
                 "option": sys_tag}
        
        off_nadir = self.off_nadir
        
        """ Read the generic data pool """
        with open(radar["source"], "r") as f:
            parameters = json.loads(f.read())
        
        pool = parameters["pool"]
        option = [x for x in pool["instruments"] 
                  if x["tag"]==sys_tag][0]
            
        """ Choose the right state vector for the altitude """
        svdata = [x for x in pool["stateVectors"] 
                  if x["tag"] == state_tag][0]
        
        """ Choose the right body """
        if svdata["frame"] in ["IAU_VENUS", "VME2000"]:
            planet = venus()
        elif svdata["frame"] in ["IAU_EARTH", "ECEF", "ICRF", "J2000"]:
            planet = earth()
        
        if svdata["type"] == 'keplerian':
            """ Convert to Cartesian state vector """
            tmp = svFromJsonArgs(svdata["elements"], planet = planet)
            mycart = [x for x in tmp["features"][0]["properties"]["stateVector"] 
                      if "IAU_" in x["frame"]][0]
            svdata["time"] = mycart["time"]
            svdata["vector"] = mycart["xyzVxVyVz"]
        
        self.v = np.linalg.norm(svdata["vector"][3:6])
        self.myAntenna = sar.getInstrument(option)
        
        self.sF = pool["sincFactor"]
        
        """ Create a perspective object """
        persp = perspective(np.array(svdata["vector"][:6]),
                            svdata["time"],
                            planet = planet)
            
        option["systemTemperature"] = (option["antennaTemperature"] + 
                                      (10**((option["noiseFigure"])/10)-1)*
                                      pool["referenceTemperature"])
        P_N = Boltzmann*option["systemTemperature"]*option["receiverBandwidth"]
        option["noisePower"] = P_N
        
        self.mygeom = persp.computeGeometry(self.off_nadir)
        self.off_nadir = np.radians(self.mygeom["off_nadir"])
        try:
            prf_range = np.linspace(*option["prf_range"])
        except KeyError:
            prf_range = None
            
        self.mydF = dF(persp, 
                       instr = self.myAntenna, 
                       prf_range = prf_range,
                       off_nadir = self.off_nadir)
        self.mydF_dict = self.mydF.fig.to_dict()
        self.option = option
        self.pool = pool
        self.radar = radar
        
        # self.echoWindowPRFs = np.arange(min(self.mydF.tP.prf_range),
        #                                 max(self.mydF.tP.prf_range), 
        #                                 10)
        self.echoWindowPRFs = self.mydF.tP.prf_range
        
        self.echoWindows = [[P.closed(*x) for x in self.mydF.echoWindows(k)] 
                            for k in self.echoWindowPRFs]
    
    @staticmethod
    def getInstrument(option):
        # #%% Compute the wavelength and beamwidths
        myInst = antenna(dutyCycle = option["dutyCycle"], 
                         antennaAz = option["azimuth"],
                         antennaEl = option["elevation"],
                         carrier = option["carrier"],
                         beamSpreadEl = option["beamSpreadEl"],
                         beamSpreadAz = option["beamSpreadAz"],
                         ohmicLoss = option["ohmicLoss"])
        return myInst

    def twoWayVoltageElevation(self, angles):
        return np.sinc(self.sF*angles/self.myAntenna.beamwidthEl)**2
    
    def twoWayVoltageAzimuth(self, angles):
        return np.sinc(self.sF*angles/self.myAntenna.beamwidthAz)**2
    
    def twoWayPowerElevation(self, angles):
        return self.twoWayVoltageElevation(angles)**2
    
    def twoWayPowerAzimuth(self, angles):
        return self.twoWayVoltageAzimuth(angles)**2
    
    def fAAR(self, frange, fp, N = 5):
        v = self.v
        wavelength = self.myAntenna.wavelength
        bands = np.arange(-N,N+1)
        amb_angles = -wavelength*fp/(2*v)*bands
        angles = wavelength*frange/(2*v)
        return [integrate.quad(self.twoWayPowerAzimuth, 
                               *tuple(angles + u_amb))[0]
                             for u_amb in amb_angles]
    
    def multilook(self, dx, fp, N=5):
        """
        Split spectrum for multilooking
        
        For a given azimuth resolution, split the measured spectrum for 
        azimuth multilooking. This splitting depends on the PRF (fp) in that
        the measured spectrum may contain regions where the distributed
        azimuth ambiguity ratio (DAAR) is poor. The value of N dictates how 
        many PRF bands should be taken into account in the computation of the
        DAAR.
        

        Parameters
        ----------
        dx : float
            Desired azimuth resolution (m).
        fp : float
            The used PRF (Hz).
        N : int, optional
            2N+1 is the number of PRF bands to consider for computation of
            the DAAR. The default is 5.

        Returns
        -------
        intervals : list(float)
            The frequency intervals for each spectral look.
        pw_sig : list(float)
            The signal power for each spectral look.
        pw_amb : list(float)
            The ambiguity power for each spectral look.
        float
            The accumulated DAAR if looks are combined in order of best
            DAAR to worst.

        """
        v = self.v
        B = min(v/dx, fp)
        nB = int(fp/B)
        frange = B*np.array([0,1])
        intervals = [frange + k*B -nB*B/2 for k in  np.arange(nB)]
        pw = np.array([self.fAAR(i, fp, N=N) for i in intervals])
        pw_tot = np.sum(pw, axis=1)
        pw_sig = pw[:,N]
        pw_amb = pw_tot - pw_sig
        daar_idx = np.argsort(pw_amb/pw_sig)
        csm_pw_sig = np.cumsum([pw_sig[k] for k in daar_idx])
        csm_pw_amb = np.cumsum([pw_amb[k] for k in daar_idx])
        return intervals, pw_sig, pw_amb, csm_pw_amb/csm_pw_sig

    def powerPerformance(self, look, fp, bands=[-2,-1,0,1,2]):
        mygeom = self.mygeom
        option = self.option
        v = self.v
        wavelength = self.myAntenna.wavelength
        
        """ Determine the reference band """
        ref_index = bands.index(0)
        
        """ Calculate the range at the off_nadir angle """
        look_range = np.interp(look, 
                               mygeom["off_nadir"],
                               mygeom["range"])
        dT = 1/fp - 2*option["pulseLength"]
        look_range_time = look_range*2/c
        echo_times = np.array([list(np.linspace(-dT/2, dT/2, 100)) 
                              + look_range_time
                              + k/fp for k in bands])
        slantRange = c/2*echo_times
        
        slantRange_ref = slantRange[ref_index, :]
        
        off_nadir_range = np.interp(slantRange, 
                              mygeom["range"], 
                              mygeom["off_nadir"], 
                              left=np.nan, 
                              right=np.nan)
        
        mygeom_ref = {
            "range": slantRange_ref,
            "off_nadir": off_nadir_range[ref_index,:],
            "swath": np.interp(slantRange_ref, 
                               mygeom["range"], 
                               mygeom["swath"]),
            "incidence": np.interp(slantRange_ref, 
                                   mygeom["range"], 
                                   mygeom["incidence"]),
            "groundLon": np.interp(slantRange_ref, 
                                   mygeom["range"], 
                                   mygeom["groundLon"])
        }
        
        twoWay = self.twoWayPowerElevation(np.radians(off_nadir_range-look))
        
        """ Compute the total power interference from range ambiguities """
        power = twoWay/slantRange**3
        power_ref = power[ref_index,:]
        power_intf = np.nansum(power, axis=0) - power_ref
        
        mygeom_ref["range ambiguity"] = 10*np.log10(power_intf/power_ref)
        
        """ Compute the azimuth ambiguity """
        u_range = [[wavelength*(-fp/2 + k*fp)/2/v, 
                    wavelength*(fp/2 + k*fp)/2/v] 
                   for k in bands] 
        azimuthPower = [integrate.quad(self.twoWayPowerAzimuth, 
                                       *tuple(u_bds))[0]
                         for u_bds in u_range]
        azimuthPower_ref = azimuthPower[ref_index]
        azimuthPower_interf = np.sum(azimuthPower) - azimuthPower_ref
        
        
        """ Compute radar equation contributors """
        P_dB = 10*np.log10(option["transmitPower"])
        P_N_dB = 10*np.log10(option["noisePower"])
        P_atm_dB = option["twoWayAtmosphericLoss"]
        mygeom_ref["QNSR"] = -6.02*option["ADCbits"]
        G_dB = self.myAntenna.gain
        NESZ_0 = 10*np.log10(4*(4*np.pi)**3*v
                             /(c*wavelength**3*option["pulseLength"]*fp))
        NESZ_cnst = NESZ_0 + P_N_dB - P_dB - 2*G_dB + P_atm_dB
        
        incidence = np.radians(mygeom_ref["incidence"])
        mygeom_ref["NESZ"] = NESZ_cnst + 10*np.log10(np.sin(incidence)
                                                     /power_ref)
        mygeom_ref["azimuth ambiguity"] = 10*np.log10(azimuthPower_interf
                                                      /azimuthPower_ref)
        mygeom_ref["MNR"] = 10*np.log10(sum([10**(mygeom_ref[amb]/10)
                                         for amb in ["azimuth ambiguity",
                                                     "range ambiguity",
                                                     "QNSR"]]))
        return pd.DataFrame(mygeom_ref)

    def performanceTraces(self, look, fp, field):
        perf = self.powerPerformance(look, fp)
    
        return [go.Scatter(
            name=key,
            x=perf[field],
            y=perf[key],
            mode='lines',
            line=dict(width=5,color=self.perf_colors[key]),
            showlegend=True
            ) for key in ["NESZ", 
                          "range ambiguity", 
                          "azimuth ambiguity", 
                          "MNR"]]

    @staticmethod
    def iLen(interval):
        if interval.empty:
            return 0
        return interval.upper - interval.lower
    
    @staticmethod
    def bracket(idx, overlap, maxIter = 50):
        """
        Find Index around a local peak
        
        Find the Index of a peak so that the morph function can compute the width
        of a peak even when the peak is flat.
    
        Parameters
        ----------
        idx : int
            Initial starting point to find peak.
        overlap : array(N,) or list
            List of function values containing peaks.
        maxIter : int, optional
            Maximum number of iterations before quitting. The default is 50.
    
        Raises
        ------
        IndexError
            Raised when no solution is found.
    
        Returns
        -------
        int
            Index of peak.
    
        """
        if maxIter == 0:
            raise IndexError
        N = len(overlap)-1
        left_idx = max([idx - 1,0])
        right_idx = min([idx + 1,N])
        if overlap[left_idx] > overlap[idx]:
            return sar.bracket(left_idx, overlap, maxIter = maxIter - 1)
        elif overlap[right_idx] > overlap[idx]:
            return sar.bracket(right_idx, overlap, maxIter = maxIter - 1)
        elif overlap[left_idx] == overlap[idx]:
            return left_idx
        else:
            return right_idx
    
    @staticmethod
    def morph(left_idx, right_idx, overlap, maxIter = 50):
        if maxIter == 0:
            raise IndexError
        N = len(overlap)-1
        new_left_idx = max([left_idx - 1,0])
        new_right_idx = min([right_idx + 1,N])
        if overlap[new_left_idx] < overlap[left_idx]:
            new_left_idx = left_idx
        if overlap[new_right_idx] < overlap[right_idx]:
            new_right_idx = right_idx
            
        if new_left_idx == left_idx and new_right_idx == right_idx:
            return left_idx, right_idx
        else:
            return sar.morph(new_left_idx, 
                             new_right_idx, 
                             overlap, 
                             maxIter = maxIter - 1)
       
    def drarSpan(self, off_nadir, prf, drar_max = -10):
        myG = self.mydF.tP.my_geometry
        perf = self.powerPerformance(off_nadir, prf)
        idxs = np.argwhere(perf["range ambiguity"]<drar_max)
        target = P.closed(perf['incidence'][idxs[0][0]], 
                          perf['incidence'][idxs[-1][0]])
        swath = np.interp([perf['incidence'][idxs[0][0]],
                           perf['incidence'][idxs[-1][0]]], 
                          myG["incidence"],
                          myG["swath"])
        rng = np.interp([perf['incidence'][idxs[0][0]],
                         perf['incidence'][idxs[-1][0]]], 
                        myG["incidence"],
                        myG["range"])
        return target, np.diff(swath)[0], np.diff(rng)[0]

    def getBestEchoWindows(self,
                           off_nadir, 
                           fp, 
                           trail = [], 
                           drar_max = -10, 
                           maxIter = 20):
        if len(trail) == maxIter:
            raise IndexError
            
        p_rng = self.mydF.tP.prf_range
        prf = p_rng[np.argmin(np.abs(p_rng - fp))]
        prfidx = np.argmin(np.abs(self.echoWindowPRFs - prf))
        target, swath, rng = self.drarSpan(off_nadir, prf, drar_max = drar_max)
        overlap = [max([sar.iLen(target.intersection(e)) for e in eW]) 
                   for eW in self.echoWindows]
        nprfidx = sar.bracket(prfidx, overlap)
        prf_span = sar.morph(nprfidx, nprfidx, overlap)
        new_prf = self.echoWindowPRFs[prf_span[0] + 
                                      (prf_span[-1]-prf_span[0])//2]
        nnew_prf = p_rng[np.argmin(np.abs(p_rng - new_prf))]
        if nnew_prf in trail:
            return nnew_prf
        else:
            trail.append(nnew_prf)
            return self.getBestEchoWindows(off_nadir, 
                                           trail[-1],
                                           trail = trail,
                                           drar_max = drar_max)
    
    def estPRF(self, off_nadir, drar_max = -10):
        myG = self.mydF.tP.my_geometry
        fp = self.mydF.tP.prf_range[0]
        maxprf = self.mydF.tP.prf_range[-1]
        r = np.interp(off_nadir, myG["off_nadir"], myG["range"])
        dF = c/2/r
        try:
            prf_list = [self.getBestEchoWindows(off_nadir, 
                                                fp, 
                                                trail = [],
                                                drar_max = drar_max)]
        except IndexError:
            return []
        
        while prf_list[-1] < maxprf:
            try:
                prf_list.append(self.getBestEchoWindows(off_nadir, 
                                                        prf_list[-1] + dF,
                                                        trail = [],
                                                        drar_max = drar_max))
            except IndexError:
                break
            if prf_list[-1] == prf_list[-2]:
                return prf_list[:-1]
        return prf_list
