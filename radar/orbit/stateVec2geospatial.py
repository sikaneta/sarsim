#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 09:04:52 2020

@author: ishuwa
"""


from measurement.measurement import state_vector_ESAEOD
from measurement.measurement import state_vector_RSO
import numpy as np
import pandas as pd
import geopandas

#%% Load up the data
#output_file = "/home/ishuwa/local/src/Python/sarsim/radar/orbit/s1b.gpkg"
output_file = "/home/ishuwa/local/src/Python/sarsim/radar/orbit/tsx.gpkg"
#orbit_file = "/home/ishuwa/local/src/Python/sarsim/radar/orbit/S1B_OPER_AUX_POEORB_OPOD_20180825T110641_V20180804T225942_20180806T005942.EOF"
orbit_file = "/home/ishuwa/local/src/Python/sarsim/radar/orbit/TSX/TDX-ORB-3-RSO+CTS-RSG_2019_063_10_00_2019_064_00_00.dat"
#sv = state_vector_ESAEOD()
sv = state_vector_RSO()
svecs = sv.readStateVectors(orbit_file)

#%% Convert data to llh
llh = [sv.xyz2polar(mD) for mD in sv.measurementData]

#%% Create the pandas data frame
df = pd.DataFrame(
    {"Latitude": [s[0] for s in llh],
     "Longitude": [s[1] for s in llh],
     "X": [x[0] for x in sv.measurementData],
     "Y": [x[1] for x in sv.measurementData],
     "Z": [x[2] for x in sv.measurementData],
     "vX": [x[3] for x in sv.measurementData],
     "vY": [x[4] for x in sv.measurementData],
     "vZ": [x[5] for x in sv.measurementData],
     "R": [s[2] for s in llh],
     "time": sv.measurementTime})

#%% Convert to geopandas
gdf = geopandas.GeoDataFrame(
    df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))

#%% Write to gpkg
gdf.to_file(output_file, layer='state_vectors', driver="GPKG")
