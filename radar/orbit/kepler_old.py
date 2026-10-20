# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:19:56 2023

@author: ishuwa.sikaneta
"""

from orbit.orientation import orbit
from measurement.measurement import state_vector
from space.planets import earth
import numpy as np
from tqdm import tqdm
import json

#%% Define tranlation dictionary for different naming convention
babel = {
    "a": ["semi_major_axis"],
    "e": ["eccentricity"],
    "ascendingNode": ["raan"],
    "epoch": ["epoch_time"],
    "arg_perigee": [],
    "inclination": [],
    "true_anomaly": []
}

def getKey(key, mydict):
    mykeys = list(mydict.keys())
    bbkeys = list(babel.keys())
    if key in mykeys:
        return mydict[key]
    if key in bbkeys:
        try:
            return [mydict[x] for x in babel[key] if x in mykeys][0]
        except IndexError:
            raise KeyError
            
#%%     
def sv2geojson(svPCR, kepler_orbit):
    stateVectors = {"type": "FeatureCollection",
                    "features": []}
    for svTime, svData in tqdm(zip(svPCR.measurementTime, svPCR.measurementData)):
        satSVME = svPCR.planet.PCRtoME2000(svTime, svData)
        satSVICRF = svPCR.planet.PCRtoICRF(svTime, svData)
        oarg, ascNode = kepler_orbit.setFromStateVector(satSVME[1])
        parg = kepler_orbit.arg_perigee
                
        mysv = {"type": "Feature",
              "geometry": {
                  "type": "Point",
                  "coordinates": list(map(svPCR.xyz2polar(svData[:3]).__getitem__, 
                                      [1,0,2]))
              },
              "properties": {
                  "version": "8",
                  "referenceDocument": "Hydroterra E12 proposal",
                  "object": "Hydroterra+",
                  "timeUTC": satSVICRF[0].astype(str),
                  "orbitAngle": np.degrees(np.arctan2(np.sin(oarg), np.cos(oarg))),
                  "argPerigee": np.degrees(np.arctan2(np.sin(parg), np.cos(parg))),
                  "semiMajorAxis": kepler_orbit.a,
                  "eccentricity": kepler_orbit.e,
                  "period": kepler_orbit.period,
                  "ascendingNode": np.degrees(kepler_orbit.ascendingNode),
                  "stateVector": [
                      {
                          "frame": "IAU_EARTH",
                          "time": svTime.astype(str),
                          "xyzVxVyVz": svData.tolist()
                      },
                      {
                          "frame": "ICRF",
                          "time": satSVICRF[0].astype(str),
                          "xyzVxVyVz": satSVICRF[1].tolist()
                      },
                      {
                          "frame": "EME2000",
                          "time": satSVME[0].astype(str),
                          "xyzVxVyVz": satSVME[1].tolist()
                      }
                  ],
                  "processingTimestamp": np.datetime64("now").astype(str)
              }
          }
        stateVectors["features"].append(mysv)
        
    return stateVectors

#%%
def svFromJsonArgs(kepler_params, planet = earth()):
    hPlus = orbit()
    k_params = {k:getKey(k, kepler_params) for k in babel.keys()}
    hPlus.setFromDict(k_params, planet=planet)
    
    sv = state_vector(planet = planet)
    if "delta_t" in kepler_params:
        prf = 1.0/kepler_params["delta_t"]
    else:
        prf = 1000
        
    if "n_samples" in kepler_params:
        N_samples = kepler_params["n_samples"]
    else:
        N_samples = 10
        
    t0 = np.datetime64(k_params["epoch"])
    dT = np.timedelta64(int(1/prf*1e9), 'ns')
    t_offset = hPlus.computeT(k_params["true_anomaly"] +
                              k_params["arg_perigee"])
    t_dT = np.timedelta64(int(t_offset*1e9), 'ns')
    time_array = t_offset + np.arange(N_samples)/prf
    orbit_angles = list(map(lambda t: hPlus.computeO(t)[0], time_array))
    sv_inertial = [(t0 + t_dT + k*dT, hPlus.computeSV(o)[0]) 
                   for k,o in enumerate(orbit_angles)]
    
    for vec in sv_inertial:
            mysv = sv.planet.ICRFtoPCR(*vec)
            sv.add(*mysv)
    
    sv_json = sv2geojson(sv, hPlus)
    
    return sv_json
    
    # with open("hplus.geojson", "w") as f:
    #     f.write(json.dumps(sv_json, indent=2))