# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:44:13 2026

@author: Ishuwa.Sikaneta
"""

from generateXML.generateConfigSimple import get_parser 
from generateXML.generateConfigSimple import genModeXML
from generateXML.generateConfigSimple import createXMLStructure
import lxml.etree as etree
import configuration.configuration as cfg
import numpy as np
import os
import utils.fileio as fio
import pandas as pd
import plotly.express as px
from scipy.constants import c
import json
import matplotlib.pyplot as plt
from omegak.cudawk import wkProcessNumbaCuda
from measurement.measurement import state_vector
from measurement.arclength import slow
import math

#%% Generate a new simulation
def_args = "--config-xml s1g.xml --beams 1 --recompute-rsys".split()
vv = get_parser().parse_args(def_args)

if os.path.abspath(vv.config_xml) != vv.config_xml:
    vv.config_xml = os.path.join(os.path.abspath(vv.sim_folder), vv.config_xml)

if vv.ridx[-1] == -1:
    vv.ridx[-1] = None
    
#%%
myxml = genModeXML(vv)

#%%
with open(vv.config_xml,'w') as outfile:
    outfile.write(etree.tostring(myxml, pretty_print=True).decode())

#%% Load the json
with open(vv.channel_json, 'r') as f:
    myjson = json.loads(f.read())
    
#%% Create the target directory if it does not exist
if vv.config_xml is not None:
    createXMLStructure(os.path.split(vv.config_xml)[0])

#%% Delete any existing pickle radar file
if not vv.keep_pickle:
    pkl_file = ".".join(vv.config_xml.split(".")[0:-1]) + "_radar.pickle"
    if os.path.exists(pkl_file):
        os.remove(pkl_file)

#%% Compute the array of radar objects
print("""Computing the radar system acquisition object""")
radar = cfg.loadConfiguration(vv.config_xml)

#%% Setup the pickled r_sys
bands = vv.bands or np.arange(-int(len(radar)/2),int(len(radar)/2)+1)

#%% Generate the r_sys object
print("""Computing the radar system signal processing object""")
if vv.recompute_rsys:
    r_sys = cfg.computeStoreRsys(radar, bands, vv.ref_rangeidx)
else:
    r_sys = cfg.loadRsys(radar) or cfg.computeStoreRsys(radar, 
                                                        bands, 
                                                        vv.ref_rangeidx)

#%% Generate the processing filter
print("""Computing the multi channel processing filter""")
if vv.recompute_Hfilter:
    H,_ = cfg.computeStoreMultiProcFilterNonLinear(radar)
else:
    H = cfg.loadMultiProcFilter(radar)
    if H is None:
        H = cfg.computeStoreMultiProcFilterNonLinear(radar)[0]
        
#%% Generate a reference ground point
myC = slow()
mySV = state_vector()
def evo(x,sat,u):
    myC.diffG(mySV.expandedState(sat))
    cfn = lambda s: sum([x*s**k/math.factorial(k) 
                         for k,x in enumerate(myC.cdf)])
    tfn = lambda s: sum([x*s**k/math.factorial(k) 
                         for k,x in enumerate(myC.cdf[1:])])
    def fn(s):
        cs = cfn(s)
        T = tfn(s)
        return np.dot(T, cs-x)/np.linalg.norm(cs-x)
    
    s = u/(fn(1)-fn(0))
    return s, cfn(s) - x

ksp = np.max(r_sys.ks[0,:]) - np.min(r_sys.ks[0,:])
pointsXYZ = list()
for idx in myjson['targets']:
    pointXYZ, satSV, satUTC = cfg.computeReferenceGroundPoint(radar, 
                                                     None, 
                                                     idx['ridx'], 
                                                     idx['xidx'])
    
    rhat = pointXYZ - satSV[:3]
    chat = satSV[:3]/np.linalg.norm(satSV[:3])
    r0 = np.linalg.norm(rhat)
    rhat /= r0
    vhat = satSV[3:6].copy()
    vhat /= np.linalg.norm(vhat)
    pointVxVyVz = np.zeros_like(pointXYZ)
    
    if "vr" in idx.keys():
        pointVxVyVz += idx['vr']*rhat
    if "vx" in idx.keys():
        pointVxVyVz += idx['vx']*vhat
    pointsXYZ.append(np.hstack((pointXYZ, pointVxVyVz)))
    idx['state'] = list(np.hstack((pointXYZ, pointVxVyVz)))
    amb_s, amb_r = evo(pointXYZ, satSV, ksp/r_sys.kr[0])
    kx_vr = -ksp*2*idx['vr']/(radar[0]['antenna']['wavelength']*r_sys.prf)
    ds_vr, _ = evo(pointXYZ, satSV, kx_vr/r_sys.kr[0])
    idx['ambiguity_ds'] = amb_s
    idx['ambiguity_rng'] = np.linalg.norm(amb_r)
    idx['ds_vr'] = ds_vr
    idx['veff'] = r_sys.vs*np.sqrt(np.linalg.norm(pointXYZ)/
                                   np.linalg.norm(satSV[:3]))
    
#%% Generate the data only for the selected channels
if vv.channel_idxs is None:
    vv.channel_idxs = range(len(radar))
    
cfg.computeSignals([radar[k] for k in vv.channel_idxs], 
                   pointsXYZ, 
                   vv.rblock_size)

#%% Generate the name of the output multi-channel processed file
output_file = fio.fileStruct(radar[0]['filename'],
                             "mchan_processed",
                             "Xr%d" % vv.ridx[0],
                             "mchanprocessed.npy")

#%% Multi-channel process the data
procData, _ = cfg.multiChannelProcessMem(radar, 
                                         vv.ridx, 
                                         p=vv.p)

#%% Write the multi-channel processed data to file
fio.writeSimFiles(output_file, 
                  procData, 
                  xblock=vv.xblock_size)

#%% Load the data
if vv.mchan_processed_file is None:
    vv.mchan_processed_file = fio.fileStruct(radar[0]['filename'],
                                             "mchan_processed",
                                             "Xr",
                                             "mchanprocessed.npy")


#%% Omega-K processed the multi-channel processed data
if vv.vanilla_stolt:
    print("Applying vanilla Stolt interpolation")
    r_sys.C.a3 = 0.0
    r_sys.C.a4 = 0.0
    

#%% Load the data
procData = fio.loadSimFiles(vv.mchan_processed_file, xidx=vv.xidx)
if not vv.no_phase_correct:
    print("Removing differential geo phase error")
    phsCorr = np.exp(-1j*r_sys.ks_phase_correction[vv.xidx[0]:vv.xidx[-1]])
    (rows,cols) = procData.shape
    for row in range(rows):
        procData[row,:] *= phsCorr[row]
        
#%% Do the SAR processing for this chunk
print("Processing")
print(vv.xidx)
mycols = [x for x in [cols, vv.rblock_size] if x is not None]
wkSignal = wkProcessNumbaCuda(procData, 
                              r_sys, 
                              ks = r_sys.ks_full[vv.xidx[0]:vv.xidx[1]],
                              os_factor=16, 
                              mem_cols = min(mycols),
                              mem_rows = 8192,
                              Mc=64)
# wkSignal = cfg.wkProcessNumba(procData, 
#                               r_sys, 
#                               ks = r_sys.ks_full[vv.xidx[0]:vv.xidx[1]],
#                               os_factor=16, 
#                               mem_cols = min(mycols),
#                               mem_rows = 8192)

#%% Call the SAR processing algorithm
if vv.wk_processed_file is None:
    vv.wk_processed_file = fio.fileStruct(radar[0]['filename'],
                                         "wk_processed",
                                         "X%dr" % vv.xidx[0],
                                         "wkprocessed.npy")
    
#%% Write the data to file
fio.writeSimFiles(vv.wk_processed_file, 
                  wkSignal, 
                  rblock=vv.rblock_size)

#%%
if 'wkSignal'not in locals():
    proc_file = fio.fileStruct(radar[0]['filename'],
                                        "wk_processed",
                                        "Xr",
                                        "wkprocessed.npy")
    print("Loading data from file: %s" % proc_file)
    wkSignal = fio.loadSimFiles(proc_file, xidx=vv.xidx, ridx=vv.ridx)
    
#%%
noiseSignal, r_sys = cfg.multiChannelNoise(radar, p=0.9)
noiseSignal = np.fft.ifft(noiseSignal)
noisePower = np.var(noiseSignal)
correction = 1.0

wkSignal = np.fft.ifft(wkSignal, axis=0)
maxSignal = np.max(np.abs(wkSignal))**2*correction
wkSignal = wkSignal/maxSignal

#%% Plot the IRFs
M,N = wkSignal.shape
dx = np.pi/np.max(r_sys.ks_full)
dr = r_sys.r[1] - r_sys.r[0]
Nbnds = len(bands)

Ns = 60
for tg in myjson['targets']:
    dn = -np.round(tg['ds_vr']/dx/Nbnds).astype(int)
    dm = np.round((r_sys.r[tg['ridx']] - tg['ambiguity_rng'])/dr).astype(int) 
    row = (max(0,Nbnds*(tg['xidx']-dn) - Ns), 
           min(Nbnds*(tg['xidx']-dn) + Ns + 1, M-1)) 
    col = (max(0,tg['ridx'] - Ns), min(tg['ridx'] + Ns + 1, N-1))
    IRF = 20*np.log10(np.abs(wkSignal[row[0]:row[1], col[0]:col[1]]))
    tg['rngMax'] = col[0] + np.argmax(np.max(IRF, axis=0))
    tg['aziMax'] = (row[0] + np.argmax(np.max(IRF, axis=1)))%M
    
    
df_azi = pd.concat(
    [pd.DataFrame(
            {
                "target": M*[x['name']], 
                "azimuth (m)": (np.arange(M)-M/2)*dx,
                "Response (dB)": 20*np.log10(np.abs(wkSignal[:,x['rngMax']]))
            }
        ) for x in myjson['targets']
    ], 
    ignore_index = True
)
    
df_rng = pd.concat(
    [pd.DataFrame(
            {
                "target": N*[x['name']], 
                "range (m)": r_sys.near_range + np.arange(N)*c/2/r_sys.fs,
                "Response (dB)": 20*np.log10(np.abs(wkSignal[x['aziMax'],:]))
            }
        ) for x in myjson['targets']
    ], 
    ignore_index = True
)

#%% Plotly
sim_folder = os.path.join(os.path.split(vv.config_xml)[0], 
                          "simulation_plots")
fig_a = px.line(df_azi, x='azimuth (m)', y='Response (dB)', color='target')
fig_a.write_html(os.path.join(sim_folder, "azimuthIRF.html"))
fig_a = px.line(df_rng, x='range (m)', y='Response (dB)', color='target')
fig_a.write_html(os.path.join(sim_folder, "rangeIRF.html"))

#%%
Ns = 200
azi = (np.arange(M)-M/2)*dx
rng = r_sys.near_range + np.arange(N)*c/2/r_sys.fs

for tg in myjson['targets']:
    drow = np.round(tg['ambiguity_ds']/dx).astype(int)
    dcol = np.round((tg['ambiguity_rng']-r_sys.r[tg['ridx']])/dr).astype(int) 
    row = (max(0,tg['aziMax']-Ns), min(tg['aziMax'] + Ns + 1, M-1)) 
    col = (max(0,tg['ridx'] - Ns), min(tg['ridx'] + Ns + 1, N-1))
    IRF = 20*np.log10(np.abs(wkSignal[row[0]:row[1], col[0]:col[1]]))
    mxIRF = np.max(IRF)
    azi_m = azi[row[0]:row[1]].copy()
    rng_m = rng[col[0]:col[1]].copy()
    rng_mean = np.mean(rng_m)
    azi_mean = np.mean(azi_m)
    fg_title = 'Impulse Response Target: %s (rng: %0.6e, azi %0.6e)' % (tg['name'],
                                                                       rng_mean,
                                                                       azi_mean)
    fig2 = px.imshow(IRF,
                     x=rng_m,
                     y=azi_m,
                     zmin = -40,
                     zmax = mxIRF + 5,
                     labels={'x':'Range (m)',
                             'y':'Azimuth (m)', 
                             'color':'Response (dB)'},
                    color_continuous_scale='RdYlGn_r',
                    title=fg_title
                    )
    fig2.write_html(os.path.join(sim_folder, "Target%s_IRF.html" % tg['name']))
    
    if tg['aziMax'] + drow < M:
        row = (max(0,tg['aziMax'] + drow - Ns), 
               min(tg['aziMax'] + drow + Ns + 1, M-1)) 
        col = (max(0,tg['ridx'] + dcol - Ns), 
               min(tg['ridx'] + dcol + Ns + 1, N-1))
        IRF = 20*np.log10(np.abs(wkSignal[row[0]:row[1], col[0]:col[1]]))
        
        azi_m = azi[row[0]:row[1]].copy()
        rng_m = rng[col[0]:col[1]].copy()
        fg_title = 'P Ambiguity Response Target: %s' % tg['name']
        fig2 = px.imshow(IRF,
                         x=rng_m,
                         y=azi_m,
                         zmin = -40,
                         zmax = mxIRF + 5,
                         labels={'x':'Range (m)',
                                 'y':'Azimuth (m)', 
                                 'color':'Response (dB)'},
                        color_continuous_scale='RdYlGn_r',
                        title=fg_title
                        )
        fig2.write_html(os.path.join(sim_folder, "Target%s_Pamb_IRF.html" % tg['name']))
        
    if tg['aziMax'] - drow > 0:
        row = (max(0,tg['aziMax'] - drow - Ns), 
               min(tg['aziMax'] - drow + Ns + 1, M-1)) 
        col = (max(0,tg['ridx'] + dcol - Ns), 
               min(tg['ridx'] + dcol + Ns + 1, N-1))
        IRF = 20*np.log10(np.abs(wkSignal[row[0]:row[1], col[0]:col[1]]))
        
        azi_m = azi[row[0]:row[1]].copy()
        rng_m = rng[col[0]:col[1]].copy()
        fg_title = 'N Ambiguity Response Target: %s' % tg['name']
        fig2 = px.imshow(IRF,
                         x=rng_m,
                         y=azi_m,
                         zmin = -40,
                         zmax = mxIRF + 5,
                         labels={'x':'Range (m)',
                                 'y':'Azimuth (m)', 
                                 'color':'Response (dB)'},
                        color_continuous_scale='RdYlGn_r',
                        title=fg_title
                        )
        fig2.write_html(os.path.join(sim_folder, "Target%s_Namb_IRF.html" % tg['name']))
        