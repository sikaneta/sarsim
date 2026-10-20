# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 09:12:29 2026

@author: Ishuwa.Sikaneta
"""

import numpy as np
import plotly.express as px
import pandas as pd
import pathlib
import utils.fileio as fio

#%% plot the space time diagram
def plotSpaceTimePlotly(N, radar, folder = None):
    if folder is None:
        folder = pathlib.Path.home()
    else:
        folder = pathlib.Path(folder)
        
    va = radar[0]['platform']['satelliteVelocity']
    number_CHANNELS = len(radar)
    
    channels = []
    for rd,k in zip(radar, range(len(radar))):
        df_ch = pd.DataFrame()
        t = np.arange(N)/rd['acquisition']['prf']
        df_ch['time (ms)'] = (t+rd['delay']['sampling'])*1e3
        df_ch['position (m)'] = (rd['delay']['baseline'] + 
                                 (t+rd['delay']['sampling'])*va)
        df_ch['channel'] = "channel-%d" % k
        df_ch['pulse'] = ["%s"%k for k in range(N)]
        channels.append(df_ch)
      
    df_chns = pd.concat(channels, ignore_index=True)
    df_chns['yproj'] = 0.0
    
    fig = px.scatter(df_chns, 
                     title = 'Space-Time Sampling Plot',
                     x='position (m)', 
                     y='time (ms)',
                     color = 'channel',
                     symbol = 'channel')
    
    fig.add_traces(px.scatter(df_chns, 
                              x='position (m)', 
                              y='yproj',
                              color = 'channel',
                              symbol = 'channel').data)
    for k in range(number_CHANNELS):
        fig.data[k].showlegend = False
    fig.write_html(folder / "spaceTimeDiagram.html")
    
def plotTwoWayPattern(H, r_sys, folder):
    if folder is None:
        folder = pathlib.Path.home()
    else:
        folder = pathlib.Path(folder)
    
    resp = list()
    D = np.sqrt(np.sum(np.abs(H)**2, axis=0))
    for idx, band in enumerate(r_sys.bands):
        df_bnd = pd.DataFrame()
        df_bnd['ks'] = r_sys.ks[idx,:]
        df_bnd['azimuth (deg)'] = np.degrees(np.arcsin(-r_sys.ks[idx,:]/r_sys.kr[0]))
        df_bnd['Response (dB)'] = 10*np.log10(D[idx,:])
        df_bnd['Band'] = "%s" % band
        resp.append(df_bnd)
        
    df_resp = pd.concat(resp, ignore_index=True)

    fig = px.scatter(df_resp, 
                     title = 'Desired Doppler Response',
                     x='azimuth (deg)', 
                     y='Response (dB)',
                     color = 'Band')
    fig.write_html(folder / "desiredResponse.html")

def plotRawData(rad, rangeidx = [0,1000], folder = None):
    if folder is None:
        folder = pathlib.Path.home()
    else:
        folder = pathlib.Path(folder)
        
    data = fio.loadNumpy_raw_dataMem([rad], 
                                     ridx = rangeidx)
    data = np.abs(np.fft.ifft(data[0,:,:], axis=1))
    
    int_data = (data/np.max(data)*256).astype(np.uint8)
    
    fig = px.imshow(int_data,
                    labels=dict(x="Azimuth (sample)", 
                                y="Range (sample)", 
                                color="Response")
                    )
    
    fig.write_html(folder / "rawData.html")