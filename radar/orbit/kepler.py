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
def svFromJsonArgs(kepler_params):
    hPlus = orbit()
    hPlus.setFromDict(kepler_params, planet=earth())
    
    sv = state_vector(planet = hPlus.planet)
    prf = 1.0/kepler_params["delta_t"]
    N_samples = kepler_params["n_samples"]
    t0 = np.datetime64(kepler_params["epoch"])
    dT = np.timedelta64(int(1/prf*1e9), 'ns')
    time_array = np.arange(N_samples)/prf
    orbit_angles = list(map(lambda t: hPlus.computeO(t)[0], time_array))
    sv_inertial = [(t0 + k*dT, hPlus.computeSV(o)[0]) 
                   for k,o in enumerate(orbit_angles)]
    
    for vec in sv_inertial:
            mysv = sv.planet.ICRFtoPCR(*vec)
            sv.add(*mysv)
    
    sv_json = sv2geojson(sv, hPlus)
    
    return sv_json
    
    # with open("hplus.geojson", "w") as f:
    #     f.write(json.dumps(sv_json, indent=2))