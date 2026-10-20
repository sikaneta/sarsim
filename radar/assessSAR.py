# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sarPayload.payload import sar
import numpy as np
from dash import Dash, html, dcc, Input, Output, get_asset_url
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import argparse

#%%
description = """
Generate performance and timing diagram for a given SAR instrument
in a given state. Performance plots at http://localhost:8050/
"""
parser = argparse.ArgumentParser(description=description)

parser.add_argument("--source",
                    help=""" Source json file with parameters for both
                             the SAR instrument and the state vectors """,
                    type=str,
                    required=True)
parser.add_argument("--sys-tag",
                    help="Tag for instrument to assess",
                    type=str,
                    required=True)
parser.add_argument("--state-tag",
                    help="Tag for state vector to use for assessment",
                    type=str,
                    required=True)

#%%
tt = " --source ../../Envision/EuroSAR/generic.json --sys-tag VenSAR --state-tag med_high"
#tt = " --source ../../Envision/EuroSAR/fromMarc.json --state-tag Hydroterra+ --sys-tag HYDROTERRA"

#%%
vv = parser.parse_args()

#%%
mySAR = sar(source = vv.source,
            sys_tag = vv.sys_tag, 
            state_tag = vv.state_tag)

#%% Plot the diamond figure
""" Define reference look """
ref_look = np.mean(np.degrees(mySAR.off_nadir))
min_look = np.degrees(np.min(mySAR.off_nadir))
max_look = np.degrees(np.max(mySAR.off_nadir))

look_range = [float("%0.1f" % k) for k in np.arange(min_look,max_look,0.1)]
look_angle_slider = html.Div(
    [
     dbc.Label("Off Nadir Angle", html_for="Look label"),
     dcc.Slider(marks={k:'' for k in look_range},
                step=None,
                id="ref_look",
                tooltip={"placement": "top", 
                         "always_visible": True},
                value=ref_look)
    ],
    className = "row"
)

prf_range = [float(prf) for prf in mySAR.mydF.tP.prf_range]
dPRF = float("%0.2f" % (prf_range[1]-prf_range[0]))
nyqIDX = np.argmin(np.abs(prf_range - mySAR.mydF.tP.dopp_BW))
prf_slider = html.Div(
    [
     dbc.Label("PRF", html_for="PRF input"),
     dcc.Slider(prf_range[0], prf_range[-1],
                step = dPRF,
                marks={prf_range[nyqIDX]: "%d" % prf_range[nyqIDX]},
                id="ref_prf",
                tooltip={"placement": "top", 
                         "always_visible": True},
                value=mySAR.mydF.tP.dopp_BW)
    ],
    className = "row"
)

""" Define a radio button group for selecting the domain """
options=['incidence', 
         'off_nadir',
         'swath',
         'range',
         'groundLon']
radio_field = html.Div(
    [
     dbc.RadioItems(
         id="domain",
         class_name="btn-group",
         inputClassName="btn-check",
         labelClassName="btn btn-outline-primary",
         labelCheckedClassName="active",
         options = [{"label": option, "value": k}
                    for k,option in enumerate(options)],
         value=0
         )
    ],
    className="radio-group"
)


form = html.Div([
    dbc.Row(
        [
            dbc.Col(radio_field),
            dbc.Col(
                dbc.FormFloating(
                    [
                        dbc.Input(id='daar', 
                                  type='number', 
                                  min=-30, 
                                  max=-8, 
                                  step=1,
                                  placeholder = "DAAR",
                                  value=-10),
                        dbc.Label("DAAR / dB")
                    ]
                )
            ),
            dbc.Col(
                dbc.FormFloating(
                    [
                        dbc.Input(id='drar', 
                                  type='number', 
                                  min=-30, 
                                  max=-8, 
                                  step=1,
                                  placeholder = "DRAR",
                                  value=-10),
                        dbc.Label("DRAR / dB"),
                    ],
                )
            ),
            dbc.Col(
                dbc.FormFloating(
                    [
                        dbc.Input(id='azresolution', 
                                  type='number', 
                                  min=float("%0.1f" % (mySAR.myAntenna.antennaAz/2)), 
                                  max=50, 
                                  step=0.1,
                                  placeholder = "Processed Azimuth Resolution",
                                  value=30),
                        dbc.Label("Az Resolution / m"),
                    ]
                )
            )
        ],
        className="g-2",
    )]
)

""" Create an instance of the app """
app = Dash(external_stylesheets=[dbc.themes.FLATLY])

""" Set the figure height """
fg_height = 1000

""" Setup the layout """
app.layout = html.Div(className='row', children=[
    html.H2("Instrument: %s" % vv.sys_tag, className="mytitle"),
    html.Img(src=get_asset_url('esaLogo.svg'), className="logo"),
    html.H2("State Vector: %s" % vv.state_tag, className='myh2'),
    dcc.Store(id="diamondData", data=mySAR.mydF_dict),
    # #radio_field,
    form,
    look_angle_slider,
    prf_slider,
    dcc.Tabs([
        dcc.Tab(
            label='Performance', 
            children=[
                dcc.Graph(id="graph0", style={"height": fg_height})
                ]
            ),
        dcc.Tab(
            label='Processing', 
            children=[
                dcc.Graph(id="graph1", style={"height": fg_height})
                ]
            )
        ]
    )
])

@app.callback(
    Output("graph0", "figure"),
    Output("graph1", "figure"),
    Output("diamondData", "data"),
    Output("ref_prf", "marks"),
    Input("domain", "value"),
    Input("ref_look", "value"),
    Input("ref_prf", "value"),
    Input("diamondData", "data"),
    Input("daar", "value"),
    Input("drar", "value"),
    Input("azresolution", "value")
)
def update_look_angle(domain_idx, 
                      off_nadir, 
                      prf, 
                      udata, 
                      mydaar, 
                      mydrar, 
                      dAz):
    """ Update the boresight angle """
    vPRF = mySAR.mydF.getValidPRFs(off_nadir)
    data = mySAR.mydF.getPRFPlotFeatures(options[domain_idx], vPRF, prf)
    pw_perf = mySAR.powerPerformance(off_nadir, 
                                    prf,
                                    dAz = dAz,
                                    daar_max = mydaar)
    
    perf = mySAR.performanceTraces(options[domain_idx], pw_perf)
    
    """ Compute and report the swath width """
    try: 
        _,drarSwath,drarRng,validSwath = mySAR.getDRARSpan(options[domain_idx],
                                                           pw_perf,
                                                           prf,
                                                           drar_max = mydrar)   
        validSwath['line']['color'] = 'darkolivegreen'
        validSwath['line']['width'] = 8
        data.append(validSwath)
    except IndexError:
        drarSwath = drarRng = 0.0
        
 
    perf[1].name = 'range ambiguity %0.1f/%0.1f km' % (drarSwath/1e3,
                                                       drarRng/1e3)
    
    """ Make the figures to plot """
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    for trace in data:
        figure.add_trace(trace, secondary_y = False)
    for trace in perf:
        figure.add_trace(trace, secondary_y = True)
        
    figure.update_layout(uirevision=options[domain_idx], height = fg_height)
    figure.update_layout(hovermode="x unified")
    figure.update_xaxes(title_text = options[domain_idx])    
    figure.update_yaxes(title_text="PRF / Hz", secondary_y=False)   
    figure.update_yaxes(title_text="Performance / dB", secondary_y=True)
    
    intv, pw_sig, pw_amb, csm_daar = mySAR.multilook(dAz, prf)
    
    aData = pd.DataFrame({
        'Compliant': 10*np.log10(csm_daar) < mydaar,
        'Number of Looks': np.arange(len(intv))+1,
        'DAAR / dB': 10*np.log10(csm_daar)
    })
    aFig = px.bar(aData, 
                  x="Number of Looks", 
                  y="DAAR / dB",
                  color='Compliant',
                  height=fg_height)
    
    mark_prfs = mySAR.estPRF(off_nadir, 
                             pw_perf,
                             drar_max = mydrar)
    prf_marks = {float(x): "%d" % x for x in mark_prfs} or None
    return figure, aFig, udata, prf_marks


if __name__ == "__main__":
    app.run(port=8050, debug=False)