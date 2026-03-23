# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 08:36:28 2024

@author: Ishuwa.Sikaneta
"""
#%%
#from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.colors as plc
from shapely import LineString, Polygon, MultiPolygon, MultiLineString
from shapely import difference
import numpy as np
# from elasticOrbit import estimateSV
from space.planets import venus
from orbit.geometry import perspective
from plot.diamond import timingPlot
import scipy.constants as cnst
import portion as P
from functools import reduce
    
#%% Timing Plots
class diamondFigure:
    def __init__(self, 
                 persp,
                 field = "incidence",
                 field_xaxis=None,
                 prf_range = None,
                 max_PRF = 3600,
                 boresight=30.0,
                 off_nadir = None,
                 mode = None,
                 figNum = 1,
                 subtitle = None,
                 showlegend=True,
                 instr = None):
        self.roi = "Undefined"
        self.cycle = 2
        self.orbitNumber = 5000
        self.mode = mode
        self.max_PRF = max_PRF
        self.timeTDB = persp.sTime
        self.perspectives = [persp]
        self.persp = self.perspectives[0]
        self.updateTitles(figNum, subtitle)
            
            
        self.targetIncidence = [30]
        self.tP = timingPlot(self.persp, 
                             instr = instr, 
                             prf_range = prf_range,
                             off_nadir = off_nadir)
        self.max_PRF = self.tP.prf_range[-1]
        self.elevBW = np.degrees(cnst.c/
                                 self.tP.instr.carrier/
                                 self.tP.instr.antennaElEff)
        self.domain = self.tP.my_geometry[field]
        self.boresight = boresight
        self.field_xaxis = field_xaxis
        
        self.nd_forbidden = self.getNadirForbiddenLines("incidence")
        self.tx_forbidden = self.getTxForbiddenLines("incidence")
        
        self.tminmax = self.tP.field2incidence("off_nadir", 
                                               [boresight-self.elevBW/2, 
                                                boresight+self.elevBW/2])
        
        self.tr_title = "Off-Nadir [%0.1f,%0.1f]" % tuple(self.tminmax)
        
        self.fig = self.plotlyFigure(field, showlegend)

        self.mypoly = MultiPolygon([Polygon([(x,y) for x,y in 
                                zip(list(tx_fb["y"]) + [tx_fb["y"][0]], 
                                    list(tx_fb["x"]) + [tx_fb["x"][0]])
                                ])
                                    for tx_fb in self.tx_forbidden])
      
    def updateTitles(self, fignum, subtitle):
        titlestr = "Fig: %d, ROI: %s, Cycle: %d, Time: %s"
        self.figTitle = titlestr % (fignum, 
                                    self.roi,
                                    self.cycle,
                                    self.timeTDB[0][:-9])
        self.subtitle = "Relations (%s)" % subtitle
        try: 
            self.fig.layout.title.text = self.computeTitle()
        except AttributeError:
            pass
        
    # def setPerspective(self, timeTDB):
    #     svdata, svtime = estimateSV(timeTDB)
    #     return perspective(sPosVel = svdata, 
    #                        sTime = svtime, 
    #                        planet = venus())
        
    
    def getForbiddenPolygons(self, 
                             times,
                             field, 
                             name, 
                             cmap,
                             rank=None,
                             showlegend=True):
        tP = self.tP
        domain = self.tP.my_geometry[field]
        
        polygons, ranks = tP.computeForbiddenPolygons(times, 
                                                      domain,
                                                      rank=rank)
        colors = plc.sample_colorscale(cmap, np.linspace(0,1,len(polygons)))
        
        return [{
            "fill": "toself",
            "line": {"color": cl, "width": 2},
            "mode": "lines",
            "name": "%s (Rank: %d)" % (name, rank),
            "showlegend": showlegend,
            "type": "scatter",
            "x": polygon[:,0],
            "y": polygon[:,1]} 
            for polygon, rank, cl in zip(polygons,ranks,colors)]
    
    def getTxForbiddenLines(self, field, showlegend=True):
        tP = self.tP
        
        return self.getForbiddenPolygons(tP.tx_forbidden, 
                                         field,
                                         "Tx",
                                         "Viridis",
                                         showlegend=showlegend)
        
    def getNadirForbiddenLines(self, field, showlegend=True):
        tP = self.tP
        return self.getForbiddenPolygons(tP.nadir_forbidden, 
                                         field,
                                         "nadir",
                                         "Oranges",
                                         rank=1,
                                         showlegend=showlegend)
        
    def getIncidencePatch(self, field, showlegend=True):
        tP = self.tP
        iminmax = tP.iminmax
        inc_title = "Incidence [%0.1f,%0.1f]" % tuple(iminmax)
        return tP.incidencePatch(iminmax[0], 
                                 iminmax[-1], 
                                 field, 
                                 name=inc_title,
                                 color="lightblue",
                                 showlegend=showlegend)
    
    def getOffNadirPatch(self, field, boresight, showlegend=True):
        tP = self.tP
        elevBW = self.elevBW
        tminmax = tP.field2incidence("off_nadir", 
                                     [boresight-elevBW/2, 
                                      boresight+elevBW/2])
        c_swath = np.interp(tminmax, 
                            tP.my_geometry["incidence"], 
                            tP.my_geometry["swath"])
        off_nadir_title = u"3dBEl %0.1f\u00B0-%0.1f\u00B0 (%0.1f km)" % tuple(list(tminmax) + [np.diff(c_swath)/1e3])
        return tP.incidencePatch(tminmax[0], 
                                 tminmax[-1],
                                 field, 
                                 name=off_nadir_title,
                                 color="rgba(144,238,144,0.3)",
                                 #color="lightgreen",
                                 #fillpattern={'shape': '-'},
                                 showlegend=showlegend)
    
    def getGroundSwathPatch(self, 
                            field, 
                            boresight, 
                            extent = 54000, 
                            showlegend=True):
        tP = self.tP
        c_swath = np.interp(boresight, 
                            tP.my_geometry["off_nadir"], 
                            tP.my_geometry["swath"])
        near_far = [c_swath - extent/2, c_swath + extent/2]
        
        
        tminmax = tP.field2incidence("swath", near_far)
        my_title = "54 km swath"
        return tP.incidencePatch(tminmax[0], 
                                 tminmax[-1],
                                 field, 
                                 name=my_title,
                                 color="rgba(147,112,219,0.3)",
                                 fillpattern={'shape': '-'},
                                 showlegend=showlegend)
    
    def getPRFOverGroundSwathPatches(self, 
                                   field, 
                                   vPRF, 
                                   extent = 54000, 
                                   showlegend=True):
        tP = self.tP
        tminmax = vPRF['incidence']
        my_title = "54 km swath"
        
        patches = [tP.incidencePatch(tminmax.lower, 
                                     tminmax.upper,
                                     field, 
                                     name=my_title,
                                     color="rgba(147,112,219,0.3)",
                                     fillpattern={'shape': '-'},
                                     showlegend=showlegend)]
        
        for PRF in vPRF['Tx_PRF']:
            if PRF.empty:
                continue
            prf_range = (PRF.lower, PRF.upper)
            prf_title = "%0.2f-%0.2f" % prf_range
            patches.append(tP.incidencePatch(tminmax.lower, 
                                             tminmax.upper,
                                             field, 
                                             name=prf_title,
                                             prf_range = prf_range,
                                             color="rgba(247,212,219,0.3)",
                                             fillpattern={'shape': '-'},
                                             showlegend=False))
            
        return patches
    
    def getNyquistLine(self, field, showlegend=True):
        tP = self.tP
        return {"x": [min(tP.my_geometry[field]), 
                      max(tP.my_geometry[field])],
                "y": [tP.dopp_BW, tP.dopp_BW],
                "showlegend": showlegend,
                "name": "Nyquist Frequency",
                "line": dict(color='black', width=2, dash='dash'),
                "mode": "lines"}
    
    def echoWindows(self, prf):
        tP = self.tP
        ln = LineString([(prf,min(tP.my_geometry["incidence"])),
                         (prf,max(tP.my_geometry["incidence"]))])
        wins = difference(ln, self.mypoly)
        
        """ Check for the geometry type since wins.geoms will not
            work for a line string """
        if wins.geom_type == 'LineString':
            wins = MultiLineString([wins])
            
        incs = [i.coords.xy[1].tolist() for i in wins.geoms]
        return incs
        
    def getPRFLine(self, field, prf, showlegend=True):
        tP = self.tP
        # ln = LineString([(prf,min(tP.my_geometry["incidence"])),
        #                  (prf,max(tP.my_geometry["incidence"]))])
        # wins = difference(ln, self.mypoly)
        # incs = [i.coords.xy[1].tolist() for i in wins.geoms]
        incs = self.echoWindows(prf)
        
        domain = tP.my_geometry[field]
        incidence = tP.my_geometry["incidence"]
        
        xPairs = [[np.interp(x, incidence, domain) for x in p] 
                  for p in incs]
        
        if field == "swath":
            xSwath = xPairs
        else:
            domain = tP.my_geometry["swath"]
            xSwath = [[np.interp(x, incidence, domain) for x in p] 
                      for p in incs]
            
        """ Compute swath lengths """
        swLen = [np.diff(x)/1e3 for x in xSwath]
        
        myPRF = [{"x": xP,
                  "y": [prf, prf],
                  "showlegend": showlegend,
                  "name": u"%0.1f\u00B0-%0.1f\u00B0 (%0.1f km)" % tuple(inc + [sw[0]]),
                  "line": dict(color='blue', 
                               width=5),
                  "mode": "lines"} for xP,inc,sw in zip(xPairs, incs, swLen)]
        
        return myPRF
    
    def getTargetIncidenceLines(self, field, showlegend=True):
        tP = self.tP
        targetIncidence = self.targetIncidence
        timeTDB = self.timeTDB
        prf_range = tP.prf_range
        domain = tP.my_geometry[field]
        incidence = tP.my_geometry["incidence"]
        
        y_min = min(prf_range)
        y_max = max(prf_range)
        
        lines = []
        for tInc, tdb in zip(targetIncidence, timeTDB):
            x_coord = np.interp(tInc, incidence, domain)
            lines.append({"x": [x_coord, x_coord],
                          "y": [y_min, y_max],
                          "showlegend": showlegend,
                          "name": "Target %0.2f" % tInc,
                          "line": dict(color='green', width=2, dash='dash'),
                          "mode": "lines"})
        return lines
    
    def getPlotFeatures(self, field, boresight, showlegend=True):
        """ Add Incidence patch rectangle to a new list of Features """
        features = [self.getIncidencePatch(field, showlegend=showlegend)]
        
        """ Add a patch showing the boresight and 3dB beamwidth """
        features.append(self.getOffNadirPatch(field, 
                                              boresight, 
                                              showlegend=showlegend))
        
        """ Add a patch showing the 54 km ground swath """
        # features.append(self.getGroundSwathPatch(field, 
        #                                          boresight, 
        #                                          showlegend=showlegend))
        
        """ Add the transmit times """
        features += self.getTxForbiddenLines(field, showlegend=showlegend)
        
        """ Add the Nadir returns """
        features += self.getNadirForbiddenLines(field, showlegend=showlegend)
        
        """ Add the Nyquist line """
        features.append(self.getNyquistLine(field, showlegend=showlegend))
        
        """ Add the target Incidence Lines """
        for line in self.getTargetIncidenceLines(field, 
                                                 showlegend=showlegend):
            features.append(line)
        
        return features
    
    def getPRFPlotFeatures(self, field, vPRF, prf, showlegend=True):
        """ Add Incidence patch rectangle to a new list of Features """
        features = [self.getIncidencePatch(field, showlegend=showlegend)]
        
        """ Add a patch showing the boresight and 3dB beamwidth """
        features.append(self.getOffNadirPatch(field, 
                                              vPRF['off_nadir'], 
                                              showlegend=showlegend))
        
        """ Add a patch showing the 54 km ground swath and valid PRFs """
        # features += self.getPRFOverGroundSwathPatches(field, vPRF, 
        #                                               showlegend=showlegend)
        
        """ Add the transmit times """
        features += self.getTxForbiddenLines(field, showlegend=False)
        
        """ Add the Nadir returns """
        features += self.getNadirForbiddenLines(field, showlegend=False)
        
        """ Add the Nyquist line """
        features.append(self.getNyquistLine(field, showlegend=showlegend))
        
        """ Add PRF line """
        features += self.getPRFLine(field, prf, showlegend=showlegend)
        
        """ Add the target Incidence Lines """
        # for line in self.getTargetIncidenceLines(field, 
        #                                          showlegend=showlegend):
        #     features.append(line)
        
        return features
    
    def computeTitle(self):
        title = """<span style="font-family: 'Times New Roman', serif; font-size: 12pt; font-weight: bold">%s</span>""" % self.figTitle 
        subtitle = """<span style="font-family: Cursive; font-size: 10pt">%s</span>""" % self.subtitle
        if self.subtitle is not None:
            title = "<br>".join([title, subtitle])
        return title
    
    def plotlyFigure(self, field, showlegend=True):
        title = self.computeTitle()
        
        layout = go.Layout(
                title=go.layout.Title(text=title),
                height=1000,
                xaxis_title=self.field_xaxis,
                yaxis_title="PRF / Hz",
                legend = dict(orientation="h",
                              yanchor="top",
                              y=-0.05,
                              xanchor="right",
                              x=1,
                              title_font_family="Times New Roman",
                              font=dict(
                                  family="Courier",
                                  size=12,
                                  color="black"
                                  ),
                              bgcolor="aliceblue",
                              bordercolor="Black",
                              borderwidth=2)
            )
    
        features = self.getPlotFeatures(field, 
                                        self.boresight, 
                                        showlegend=showlegend)
    
        return go.Figure(layout = layout, 
                         data = features)
    
    def getValidPRFs(self, boresight, extent=54000):
        """ Check whether the boresight/PRF combination is allowed """
        tP = self.tP
        c_swath = np.interp(boresight, 
                            tP.my_geometry["off_nadir"], 
                            tP.my_geometry["swath"])
        near_far = [c_swath - extent/2, c_swath + extent/2]
        
        tminmax = tP.field2incidence("swath", near_far)
        f_points = [[prf for inc,prf in zip(forbidden["x"], forbidden["y"]) 
                     if tminmax[0] <= inc <= tminmax[-1]] 
                    for forbidden in self.tx_forbidden]
        
        out_of_range_PRF = P.closed(np.min(tP.prf_range),
                                    np.max(tP.prf_range)).complement()
        
        PRF_ok = P.closed(tP.dopp_BW, self.max_PRF)
        
        tx_forbidden_sets = [P.closed(min(x),max(x)) for x in f_points if x]
        tx_forbidden_sets = reduce(lambda x,y: x|y, tx_forbidden_sets, 
                                   out_of_range_PRF)
        tx_allowed = PRF_ok.intersection(tx_forbidden_sets.complement())
        
        f_points = [[prf for inc,prf in zip(forbidden["x"], forbidden["y"]) 
         if tminmax[0] <= inc <= tminmax[-1]] 
              for forbidden in self.nd_forbidden]
        
        nd_forbidden_sets = [P.closed(min(x),max(x)) for x in f_points if x]
        nd_forbidden_sets = reduce(lambda x,y: x|y, nd_forbidden_sets, 
                                   out_of_range_PRF)
        nd_allowed = PRF_ok.intersection(nd_forbidden_sets.complement())
        tIncs = [P.singleton(x) for x in self.targetIncidence]
        return {
            "off_nadir": boresight,
            "swath_center": c_swath,
            "extent": near_far,
            "targetIncidence": reduce(lambda x,y: x|y, tIncs),
            "incidence": P.closed(*tminmax),
            "Tx_PRF": tx_allowed,
            "Nadir_PRF": nd_allowed
            }
    
    def setBoresight(self, boresight):
        self.boresight = boresight
        
        self.tminmax = self.tP.field2incidence("off_nadir", 
                                               [boresight-self.elevBW/2, 
                                                boresight+self.elevBW/2])
        
        new_trace = self.tP.incidencePatch(self.tminmax[0], 
                                           self.tminmax[-1], 
                                           self.domain)
        
        self.fig.update_traces(x=new_trace['x'], 
                               selector=dict(name=self.tr_title))
        self.tr_title = "Off-Nadir [%0.1f,%0.1f]" % tuple(self.tminmax)
 
