# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 09:31:07 2023

@author: ishuwa.sikaneta
"""

import pandas as pd
import numpy as np
import json

#%% Define the file location
excelFile = r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\Envision\ROIs\Copy of longitudeBands-updated-bySST-step5.xlsx"

#%% Read the excel file
df = pd.read_excel(excelFile, sheet_name = "B1-ROI-Corner-pts_All_Obs")

#%%
def getROICoordinates(df, roi_number):
    roi = df[df['ROI_No']==roi_number]
    crds = [[row.X_dd, row.Y_dd, 0] 
            for index, row in roi.sort_values("Corner_ORDER").iterrows()]
    if crds[0] != crds[-1]:
        crds.append(crds[0])
        
    """ Check for standard SAR """
    sd = [not roi[roi['C%d_Std' % k].str.lower()=='sd'].empty for k in range(1,7)]
         
    """ Check for stereo SAR """
    st = [not roi[roi['C%d_Std' % k].str.lower()=='st'].empty for k in range(1,7)]
         
    """ Check for polarimetry SAR """
    sp = [not roi[roi['C%d_Std' % k].str.lower()=='sp'].empty for k in range(1,7)]
         
    """ Check for High-Res SAR """
    hr = [not roi[roi['C%d_HR' % k].str.lower()=='hr'].empty for k in range(1,7)]
    
    plan = {
        "type": roi["Obs_Type"].iloc[0],
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


for roi_number in df["ROI_No"].unique():
    crds, plan = getROICoordinates(df, roi_number)
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [crds]
            },
        "properties": {
            "ROI_No": str(roi_number),
            "plan": plan
            }
        }
    roiset["features"].append(feature)
    
#%% write to file
with open(r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\Envision\ROIs\roi.geojson", "w") as f:
    f.write(json.dumps(roiset, indent=2))
