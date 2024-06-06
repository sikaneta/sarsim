# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:17:09 2023

@author: ishuwa.sikaneta
"""

#%%
areSat = [
        {
            "excelfile": r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\RoseL\ROSE-UM-ADSF-SAR-1001590382_Iss.01_Antenna_Model_Installation\ROSEL_AntennaModel\Inputs\ROSEL_DPNear_mid.xlsx",
            "tag": "roseL",
            "array": {
                "excel": {
                    "sheet": "SystemParameters",
                    "header": [0],
                    "index_col": [0],
                },
                "frequency": {"value": "Central frequency", "field": "Value"},
                "nElementsAzimuth": {"value": 60},
                "nElementsElevation": {"value": "Antenna rows (rg)", "field": "Value"},
                "elevationLength": {"value": "Antenna size range", "field": "Value"},
                "azimuthLength": {"value": "Antenna size azimuth", "field": "Value"},
                "nominalTRMPower": {"value": "Central frequency", "field": "Value"}
                },
            "elementPattern": {
                "excel": {
                    "sheet": "patternElement",
                    "header": [0],
                    "index_col": [0],
                    },
                "u_col": "u",
                "v_col": "v",
                "elevation": ["elevation%sP_abs / 1",
                                 "elevation%sP_phase / deg"],
                "azimuth": ["azimuth%sP_abs / 1",
                            "azimuth%sP_phase / deg"]
                },
            "RF": {
                "excel": {
                    "sheet": "SystemParameters",
                    "header": [0],
                    "index_col": [0],
                },
                "frequency": {"value": "Central frequency", "field": "Value"},
                "power": {"value": "T/R modules nominal peak rf power", "field": "Value"},
                "losses": {"value": "Front end losses", "field": "Value"},
                "noiseFigure": {"value": "System figure noise", "field": "Value"},
                },
            "mode": {
                "excel": {
                    "sheet": "DPN",
                    "header": [0],
                    "index_col": [0],
                },
                "prf": {"value": "PRF", "field": "%d"},
                "pulseBandwidth": {"value": "Bandwidth", "field": "%d"},
                "pulseDuration": {"value": "Pulse Length", "field": "%d"},
                "fs": {"value": "Sampling frequency", "field": "%d"},
                "burstDuration": {"value": "Burst duration", "field": "%d"},
                "numRangeSamples": {"value": "Range lines", "field": "%d"},
                "nearRange": {"value": "Slant range near", "field": "%d"},
                "SWST": {"value": "SWST", "field": "%d"},
                },
            "excitation": {
                "channels":{
                    "excel": {
                        "sheet": "ExcitationCoeffs",
                        "header": [0,1],
                        "index_col": [0,1,2]
                        },
                    "tx": ["Transmit gain", "Transmit phase"],
                    "rx": ["Receive gain", "Receive phase"]
                },
                "preCombine": {
                    "excel": {
                        "sheet": "ExcitCoeffError",
                        "header": [0],
                        "index_col": [0]
                    },
                    "sizeTx": [12,60],
                    "sizeRx": [12,60],
                    "colTaper": ["Elements Taper %s-pol abs",
                                 "Elements Taper %s-pol phase"]
                }
            },
            "orbit": {
                "type": "StateVector",
                "frame": "ECEF",
                "time": "2011-09-01T17:59:27.192398",
                "xyzVxVyVz": [7.0768589e6,
                              1.7539093e4,
                              -0,
                              -4.939,
                              -1.583896e3,
                              7.430335e3]
                }
        },
        {
            "tag": "S1NG",
            "excitation": {
                "sheet": "ExcitationCoeffs",
                "header": [0,1],
                "index": [0,1,2],
                "tx": ["Transmit gain", "Transmit phase"],
                "rx": ["Receive gain", "Receive phase"],
            },
            "elementPattern": {
                "sheet": "Antenna1D",
                "header": [0],
                "skiprows": [1],
                "u_col": "u",
                "v_col": "v",
                "elevation": ["ElmeanAmplitude%s"],
                "azimuth": ["AZmeanAmplitude%s"]
            },
            "orbit": {
                "type": "StateVector",
                "frame": "ECEF",
                "time": "2011-09-01T17:59:27.192398",
                "xyzVxVyVz": [7.0768589e6,
                              1.7539093e4,
                              -0,
                              -4.939,
                              -1.583896e3,
                              7.430335e3]
                }
        }
]