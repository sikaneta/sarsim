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


#%%
altitude = "medium"
sysOption = "EuroSAR3_2small"
mySAR = sar(source = r"sarPayload/generic.json",
            sysOption = sysOption, 
            altitude = altitude)

#%% Plot the diamond figure
""" Define reference look """
ref_look = 30.0

""" Create the Dash layout and place items """
app = Dash(__name__,
           external_stylesheets=[dbc.themes.SOLAR])

look_range = [float("%0.1f" % k) for k in np.arange(20,40,0.1)]
look_angle_slider = html.Div(
    [
     dbc.Label("Off Nadir Angle", html_for="Look label"),
     dcc.Slider(marks={k:'' for k in look_range},
                step=None,
                id="ref_look",
                tooltip={"placement": "top", 
                         "always_visible": True},
                value=30)
    ],
    className = "row"
)

prf_range = mySAR.mydF.tP.prf_range
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
app = Dash(external_stylesheets=[dbc.themes.SPACELAB])

""" Setup the layout """
app.layout = html.Div(className='row', children=[
    html.H2("Diamond Timing Diagram: %s" % sysOption, className="mytitle"),
    html.Img(src=get_asset_url('esaLogo.svg'), className="logo"),
    html.H2("Altitude: %s" % altitude, className='myh2'),
    dcc.Store(id="diamondData", data=mySAR.mydF_dict),
    #radio_field,
    form,
    look_angle_slider,
    prf_slider,
    dcc.Tabs([
        dcc.Tab(
            label='Performance', 
            children=[
                dcc.Graph(id="graph0")
                ]
            ),
        dcc.Tab(
            label='Processing', 
            children=[
                dcc.Graph(id="graph1")
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
def update_look_angle(domain_idx, off_nadir, prf, data, mydaar, mydrar, dAz):
    """ Update the boresight angle """
    vPRF = mySAR.mydF.getValidPRFs(off_nadir)
    data = mySAR.mydF.getPRFPlotFeatures(options[domain_idx], vPRF, prf)
    perf = mySAR.performanceTraces(off_nadir, prf, options[domain_idx])
    
    """ Compute and report the swath width """
    try: 
        _,drarSwath,drarRng = mySAR.drarSpan(off_nadir, 
                                             prf, 
                                             drar_max = mydrar)
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
        
    figure.update_layout(uirevision=options[domain_idx], height = 600)
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
                  height=600)
    
    mark_prfs = mySAR.estPRF(off_nadir, drar_max = mydrar)
    prf_marks = {x: "%d" % x for x in mark_prfs} or None
    return figure, aFig, data, prf_marks


if __name__ == "__main__":
    app.run(port=8050, debug=False)