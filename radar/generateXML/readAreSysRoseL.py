#!/usr/bin/env python2.7

#%%
""" This python program should generate the config files for burst mode multi 
    or single channel data.
    The program generates an xml file to be processed in MATLAB """

import numpy as np
import json
from scipy.constants import c
import pandas as pd
import os

from orbit.orientation import orbit as kepler_orbit
from orbit.geometry import getTiming
from measurement.measurement import state_vector
from utils.cjson import dict2json, mtx2svddict
from generateXML.areSysSystems import areSat


#%%
def cxField(frm, 
            keys, 
            pol = None,
            angleUnits = "degrees",
            gainUnits = "linear",
            default_missing = 0.0):
    if pol is not None:
        mykeys = [k % pol for k in keys]
    try:
        Abs = frm[mykeys[0]].to_numpy()
        if gainUnits == "dB":
            Abs = 10**(Abs/10)
        if len(keys) > 1:
            Phs = frm[mykeys[1]].to_numpy()
            if angleUnits == "degrees":
                Phs = np.radians(Phs)
            return Abs*np.exp(1j*Phs)
        else:
            return Abs
    except KeyError:
        return default_missing

#%%
def cxFieldLoc(frm, 
               keys, 
               pol = None,
               angleUnits = "degrees",
               gainUnits = "linear",
               default_missing = 0.0,
               locKey = "Value"):
    if pol is not None:
        mykeys = [k % pol for k in keys]
    try:
        Abs = np.array(eval(frm.loc[mykeys[0], locKey]))
        if gainUnits == "dB":
            Abs = 10**(Abs/10)
        if len(keys) > 1:
            Phs = np.array(eval(frm.loc[mykeys[1], locKey]))
            if angleUnits == "degrees":
                Phs = np.radians(Phs)
            return Abs*np.exp(1j*Phs)
        else:
            return Abs
    except KeyError:
        return default_missing

#%%
def get(df, item, field=None):
    if "field" in item.keys():
        if field is None:
            return df.loc[item["value"]][item["field"]]
        else:
            return df.loc[item["value"]][field]
    else:
        return item["value"]
    
#%% Read the antenna positions
def defineArray(pool):
    dct = pool["array"]
    excel = dct["excel"]
    df = pd.read_excel(pool["excelfile"],
                       index_col = excel["index_col"],
                       sheet_name = excel["sheet"])
    nEle = get(df, dct["nElementsElevation"])
    nAzi = get(df, dct["nElementsAzimuth"])
    sEle = get(df, dct["elevationLength"])
    sAzi = get(df, dct["azimuthLength"])
    return {
        "type": "planar array",
        "units": "m",
        "elevationPositions": np.arange(nEle)*sEle/nEle - sEle*(1-1/nEle)/2,
        "azimuthPositions": np.arange(nAzi)*sAzi/nAzi - sAzi*(1-1/nAzi)/2
    }

#%%
def getElementPattern(pool):
    epat = pool["elementPattern"]
    elementPattern = pd.read_excel(pool["excelfile"], 
                                   sheet_name = epat["excel"]["sheet"],
                                   header = epat["excel"]["header"])
    
    try:
        u = elementPattern["u"].to_numpy()
        v = elementPattern["v"].to_numpy()
        elevH = cxField(elementPattern, keys=epat["elevation"], pol="H")
        azimH = cxField(elementPattern, keys=epat["azimuth"], pol="H")
        elevV = cxField(elementPattern, keys=epat["elevation"], pol="V")
        azimV = cxField(elementPattern, keys=epat["azimuth"], pol="V")
        return {
            "u": u, 
            "v": v, 
            "H": {
                "azimuth": azimH, 
                "elevation": elevH
            },
            "V": {
                "azimuth": azimV, 
                "elevation": elevV
            }
        }
    except KeyError:
        print("Pattern not in spreadsheet. Please check")
        return {}
    
#%%
def defineSystem(pool):
    dct = pool["RF"]
    excel = dct["excel"]
    df = pd.read_excel(pool["excelfile"],
                       index_col = excel["index_col"],
                       sheet_name = excel["sheet"])
    return {
        "carrierFrequency": get(df, dct["frequency"])*1e9,
        "array": defineArray(pool),
        "elementPattern": getElementPattern(pool),
        "losses": get(df, dct["losses"]),
        "power": get(df, dct["power"])
    }
    
#%%
def antennaWeights(pool, tPol="H", rPol="H", swathNum=1, azimuthChannel = 0):
    levelB = pool["excitation"]["channels"]
    """ Fix the following so that we haven't hard coded 13 """
    dd = pd.read_excel(pool["excelfile"], 
                       sheet_name = levelB["excel"]["sheet"], 
                       header = levelB["excel"]["header"],
                       na_filter = False,
                       index_col = levelB["excel"]["index_col"])
    
    lindex = dd.index.tolist()
    lindex = [(l[0], 1, l[-1]) for l in lindex if l[1]==''] + [l for l in lindex if l[1] !='']
    repeated = list(set([z[0] for z in [[y for y in lindex if y == x] 
                                        for x in lindex] if len(z)>1]))
    for rp in repeated:
        lindex[lindex.index(rp)] = (rp[0], rp[1], None)
        
    dd.index = lindex
    
    levelA = pool["excitation"]["preCombine"]
    subarray = pd.read_excel(pool["excelfile"], 
                             sheet_name = levelA["excel"]["sheet"], 
                             header = levelA["excel"]["header"],
                             index_col = levelA["excel"]["index_col"])
    
    azPreCombineTx = cxFieldLoc(subarray, 
                                levelA["colTaper"], 
                                pol=tPol, 
                                gainUnits='dB')
    
    azPreCombineRx = cxFieldLoc(subarray, 
                                levelA["colTaper"], 
                                pol=rPol, 
                                gainUnits='dB')
    
    mykeys = [x for x in dd.keys() if type(x[1])==int]
    qindex = [x for x in dd.index if x[1]==swathNum and x[-1] is not None]
    txGainKeys = [k for k in mykeys if k[0]==levelB["tx"][0]]
    txPhaseKeys = [k for k in mykeys if k[0]==levelB["tx"][1]]
    rxGainKeys = [k for k in mykeys if k[0]==levelB["rx"][0]]
    rxPhaseKeys = [k for k in mykeys if k[0]==levelB["rx"][1]]
    TX = (dd.loc[qindex, txGainKeys].to_numpy()*
          np.exp(1j*np.radians(dd.loc[qindex, txPhaseKeys].to_numpy())))
    RX = (dd.loc[qindex, rxGainKeys].to_numpy()*
          np.exp(1j*np.radians(dd.loc[qindex, rxPhaseKeys].to_numpy())))
    TX = TX.astype(np.complex128)
    RX = RX.astype(np.complex128)
    
    rx_channel = np.zeros((RX.shape[1]))
    rx_channel[azimuthChannel] = 1
    
    # tx = np.kron(azPreCombineTx, TX)
    # rx = np.kron(azPreCombineRx, RX.dot(np.diag(rx_channel)))
    tx = np.kron(TX, azPreCombineTx)
    rx = np.kron(RX.dot(np.diag(rx_channel)), azPreCombineRx)
    return {
        "tx": {
            "polarization": tPol,
            "weights": mtx2svddict(tx)
        },
        "rx": {
            "polarization": rPol,
            "weights": mtx2svddict(rx)
        }
    }

#%%
def getMode(pool, 
            azStartTime="2015-01-01T17:00:00.000000", 
            tPol="H", 
            rPol="H", 
            swathNum=1,
            azimuthChannel=0):
    mdict = pool["mode"]
    mode = pd.read_excel(pool["excelfile"], 
                         index_col = mdict["excel"]["index_col"],
                         header = mdict["excel"]["header"],
                         sheet_name = mdict["excel"]["sheet"])
    prf = mode.loc[mdict["prf"]["value"], swathNum]
    burst_len = mode.loc[mdict["burstDuration"]["value"], swathNum]
    return {
        "acquisition": {
            "startTime": azStartTime,
            "prf": prf,
            "numAzimuthSamples": int(prf*burst_len),
            "numRangeSamples": int(get(mode, mdict["numRangeSamples"], swathNum)),
            "nearRangeTime": get(mode, mdict["nearRange"], swathNum)*1e3*2/c,
            "rangeSampleSpacing": 1/get(mode, mdict["fs"], swathNum)*1e-6*c/2
        },
        "chirp": {
            "pulseBandwidth": get(mode, mdict["pulseBandwidth"], swathNum)*1e6,
            "length": get(mode, mdict["pulseDuration"], swathNum)*1e-6
        },
        "radarConfiguration": antennaWeights(pool, 
                                             tPol=tPol,
                                             rPol=rPol,
                                             swathNum=swathNum,
                                             azimuthChannel=azimuthChannel)
    }

#%%
tPool = areSat[0]
tPol="H"
rPol="H"
azStartTime="2015-01-01T17:00:00.000000"
pool = {
    "instrument": defineSystem(tPool),
    "platform": {"orbit": tPool["orbit"]},
    "signalData": [getMode(tPool, 
                   azStartTime=azStartTime,
                   tPol=tPol,
                   rPol=rPol,
                   swathNum=1,
                   azimuthChannel=k) for k in range(5)]
}

#%%
myjson = dict2json(pool)

#%%
folder = os.path.join(r"C:\Users",
                      r"ishuwa.sikaneta",
                      r"OneDrive - ESA",
                      r"Documents",
                      r"ESTEC",
                      r"RoseL")
filename = "RoseL_Swath1_HH_IW.json"
with open(os.path.join(folder, filename), "w") as f:
    f.write(json.dumps(myjson, indent=2))

