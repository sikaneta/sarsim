# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 09:31:07 2023

@author: ishuwa.sikaneta
"""

import pandas as pd
import numpy as np
import json
import os

#%% Define the file location
txtFile = r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\Envision\ROIs\PlanB\ESA-ENVIS-EST-MIS-ML-005_longitudeBands_forSST_9Jun2023_B1-Corner-points-and-Obs-codes_JL.txt"

#%% Read the excel file
with open(txtFile, "r") as f:
    lines = [x.split() for x in f.read().split('\n')[1:-1]]
    
rois = [[lines[k+i] for i in range(5)] for k in range(0, len(lines), 5)] 

#%%
def getROICoordinates(roi_list):
    roi = roi_list[0][0]
    crds = [[float(row[3]), float(row[4]), 0.0] for row in roi_list[1:]]
    if crds[0] != crds[-1]:
        crds.append(crds[0])
        
    """ Check for standard SAR """
    obsPlan = roi_list[0][-1].upper()
    sd = ['C%d-SD' % k in obsPlan for k in range(1,7)]
    st = ['C%d-ST' % k in obsPlan for k in range(1,7)]
    sp = ['C%d-SP' % k in obsPlan for k in range(1,7)]
    hr = ['C%d-HR' % k in obsPlan for k in range(1,7)]

    plan = {
        "type": roi_list[0][-1].split('_')[0],
        "standard": [k+1 for k,val in enumerate(sd) if val],
        "polarimetry": [k+1 for k,val in enumerate(sp) if val],
        "stereo": [k+1 for k,val in enumerate(st) if val],
        "highRes": [k+1 for k,val in enumerate(hr) if val]
    }
    filt_plan = {k:v for k,v in plan.items() if v}
    return crds, filt_plan

#%%
roiset = {
    "type": "FeatureCollection",
    "features": []
    }


for roi in rois:
    crds, plan = getROICoordinates(roi)
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [crds]
            },
        "properties": {
            "ROI_No": roi[0][0],
            "plan": plan
            }
        }
    roiset["features"].append(feature)
    
#%% write to file
with open(r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\Envision\ROIs\PlanB\roi.geojson", "w") as f:
    f.write(json.dumps(roiset, indent=2))
