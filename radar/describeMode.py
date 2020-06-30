#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 18:40:13 2020

@author: ishuwa
"""


from scipy.constants import c
import xml.etree.ElementTree as etree
import argparse
import numpy as np
import os

#%% Argparse stuff
parser = argparse.ArgumentParser(description="Describe parameters of a mode")

parser.add_argument("--config-xml", help="The config XML file", 
                    default = u'/home/ishuwa/simulation/10cm/10cm.xml')
parser.add_argument("--pulse-duration",
                    help="Specify a pulse duration in us",
                    type=float,
                    default=50)

vv = parser.parse_args()

#%% Load the radar configuration
pool = etree.parse(vv.config_xml).getroot()

#%% Get some good stuff
prfs = pool.findall(".//pulseRepetitionFrequency")
azlen = pool.find(".//azimuthElementLengths")
pulseBW = pool.find(".//pulseBandwidth")
pulseDuration = pool.find(".//pulseDuration")
carrier = pool.find(".//carrierFrequency")
#%%
this_nchan = len(prfs)
this_M = int(np.sqrt(this_nchan) - 1)
this_prf = float(prfs[0].text)*(this_M+1)
this_totallen = np.sum(np.array([float(x) for x in azlen.text.split()]))
this_subaperturelen = this_totallen/(this_M+1)
this_pulseBW = float(pulseBW.text)
this_swath = 1.0/this_prf*c/2*0.9
this_pulseDuration = vv.pulse_duration or float(pulseDuration.text)
this_swath = (1.0/this_prf - 2*this_pulseDuration*1e-6)*c/2*0.9/1e3
this_carrier = float(carrier.text)/1e9
this_mode = os.path.split(vv.config_xml)[-1].split(".")[0].replace("cm", " cm")

fmt_str = "{\\bf %s} & %0.2f & %0.1f & %0.1f & %d & %0.1f & %0.2f & %0.2f\\\\\hline"
print(fmt_str % (this_mode,
                 this_prf,
                 this_totallen,
                 this_subaperturelen,
                 this_M+1,
                 this_swath,
                 this_carrier,
                 this_pulseBW))